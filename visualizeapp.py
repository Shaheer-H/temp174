from flask import Flask, request, render_template, jsonify
import os
from model import DigitNet, OperatorNet
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), UPLOAD_FOLDER)

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Determine device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
digit_net = DigitNet().to(device)
operator_net = OperatorNet().to(device)

# Define class mappings
operator_mapping = {
    0: '+',    # add
    1: '.',    # dec
    2: '÷',    # div
    3: '=',    # eq
    4: '×',    # mul
    5: '-',    # sub
    6: 'x',    # variable x
    7: 'y',    # variable y
    8: 'z'     # variable z
}

# Load trained weights with CPU fallback
try:
    digit_net.load_state_dict(torch.load('weights/digit_net_best.pth', 
                                        map_location=device,
                                        weights_only=True))
    operator_net.load_state_dict(torch.load('weights/operator_net_best.pth', 
                                           map_location=device,
                                           weights_only=True))
    print("Model weights loaded successfully")
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise

# Set models to evaluation mode
digit_net.eval()
operator_net.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def segment_symbols(image_path: str, output_dir: str = 'debug_output') -> list[np.ndarray]:
    # Read and preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean noise
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Extract and sort symbols
    components = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 20:
            continue
        symbol = binary[y:y+h, x:x+w]
        components.append({'index': i, 'x': x, 'y': y, 'symbol': symbol})

    row_threshold = np.mean([stats[i][3] for i in range(1, num_labels)]) / 2
    
    # Sort left to right
    rows = []
    current_row = [components[0]]
    
    for comp in components[1:]:
        if abs(comp['y'] - current_row[0]['y']) < row_threshold:
            current_row.append(comp)
        else:
            rows.append(sorted(current_row, key=lambda c: c['x']))
            current_row = [comp]
    rows.append(sorted(current_row, key=lambda c: c['x']))
    
    # Resize symbols to model input size (e.g., 28x28 for MNIST)
    idx = 1
    processed_symbols = []
    for row in sorted(rows, key=lambda r: r[0]['y']):
        for comp in row:
            resized = cv2.resize(comp['symbol'], (28, 28), interpolation=cv2.INTER_AREA)
            processed_symbols.append(resized)
            cv2.imwrite(os.path.join(output_dir, f'symbol_{idx}_y{comp["y"]}_x{comp["x"]}.png'), resized)
            idx += 1
    
    return processed_symbols

def process_single_symbol(image_array, transform, device):
    """Process a single symbol image array"""
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    image = image.convert('L')
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def get_prediction(image_tensor, digit_net, operator_net, operator_mapping):
    """Get prediction for a single symbol"""
    with torch.no_grad():
        digit_output = digit_net(image_tensor)
        operator_output = operator_net(image_tensor)
        
        digit_probs = torch.nn.functional.softmax(digit_output, dim=1)
        operator_probs = torch.nn.functional.softmax(operator_output, dim=1)
        
        digit_conf, digit_pred = torch.max(digit_probs, 1)
        operator_conf, operator_pred = torch.max(operator_probs, 1)
        
        if digit_conf > operator_conf:
            return str(digit_pred.item()), float(digit_conf.item())
        else:
            operator_idx = operator_pred.item()
            return operator_mapping[operator_idx], float(operator_conf.item())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Segment the image into individual symbols
            symbol_images = segment_symbols(filepath, 'debug_output')
            
            if not symbol_images:
                return jsonify({'error': 'No symbols detected in image'})
            
            # Process each symbol
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            detected_symbols = []
            confidences = []
            
            for symbol_image in symbol_images:
                # Convert to uint8 if not already
                if symbol_image.dtype != np.uint8:
                    symbol_image = (symbol_image * 255).astype(np.uint8)
                image_tensor = process_single_symbol(symbol_image, transform, device)
                symbol, confidence = get_prediction(image_tensor, digit_net, operator_net, operator_mapping)
                detected_symbols.append(symbol)
                confidences.append(confidence)
            
            # Build equation string
            equation = ' '.join(detected_symbols)
            
            # Try to evaluate the equation
            try:
                calc_eq = equation.replace('×', '*').replace('÷', '/')
                if '=' in calc_eq:
                    solution = "Equations not supported yet"
                else:
                    try:
                        result = eval(calc_eq)
                        solution = f"Result: {result}"
                    except:
                        solution = "Building equation: " + equation
            except:
                solution = "Building equation: " + equation
            
            # Clean up
            os.remove(filepath)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences)
            
            return jsonify({
                'equation': equation,
                'solution': solution,
                'confidence': f"{avg_confidence * 100:.2f}%",
                'num_symbols': len(detected_symbols)
            })
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return jsonify({'error': str(e)})
        
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    print("Starting Flask app...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)
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

def segment_symbols(image_path):
    """Segment image into individual symbols"""
    # Read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding and noise removal
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours left-to-right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    symbol_images = []
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out noise (very small contours)
        if w * h < 100:  # Adjust this threshold based on your images
            continue
            
        # Extract symbol
        symbol = gray[y-5:y+h+5, x-5:x+w+5]  # Add padding
        if symbol.size == 0:
            continue
            
        # Convert to PIL Image
        symbol_pil = Image.fromarray(symbol)
        symbol_images.append(symbol_pil)
    
    return symbol_images

def process_single_symbol(image, transform, device):
    """Process a single symbol image"""
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
            symbol_images = segment_symbols(filepath)
            
            if not symbol_images:
                return jsonify({'error': 'No symbols detected in image'})
            
            # Process each symbol
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            detected_symbols = []
            confidences = []
            
            for symbol_image in symbol_images:
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
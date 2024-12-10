from flask import Flask, request, render_template, jsonify
import os
from model import ImprovedNet
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
digit_net = ImprovedNet(num_classes=10).to(device)    # 10 classes for digits
operator_net = ImprovedNet(num_classes=9).to(device)  # 9 classes for operators

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

# Load trained weights
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
    # Read and preprocess image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Get all components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    # Debug: Print all component stats
    print("\nAll Components:")
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect = w/h if h > 0 else 0
        print(f"Component {i}: x={x}, y={y}, w={w}, h={h}, area={area}, aspect={aspect:.2f}")
    
    # Separate horizontal lines and regular components with more lenient criteria
    horizontal_lines = []
    regular_components = []
    min_area = 5
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
            
        # More lenient horizontal line criteria
        aspect_ratio = w / h if h > 0 else 0
        if (aspect_ratio > 2 and h <= 15) or (w > 15 and h <= 5):  # Relaxed criteria
            print(f"Found horizontal line: Component {i} with aspect ratio {aspect_ratio:.2f}")
            horizontal_lines.append({'id': i, 'x': x, 'y': y, 'w': w, 'h': h})
        else:
            regular_components.append({'id': i, 'x': x, 'y': y, 'w': w, 'h': h})
    
    # Debug: Draw all components before merging
    debug_pre = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for comp in horizontal_lines:
        x, y, w, h = comp['x'], comp['y'], comp['w'], comp['h']
        cv2.rectangle(debug_pre, (x,y), (x+w,y+h), (0,0,255), 1)  # Red for horizontal lines
    for comp in regular_components:
        x, y, w, h = comp['x'], comp['y'], comp['w'], comp['h']
        cv2.rectangle(debug_pre, (x,y), (x+w,y+h), (0,255,0), 1)  # Green for regular
    cv2.imwrite('debug_output/pre_merge.png', debug_pre)
    
    # Merge horizontal lines into equals signs with more lenient criteria
    merged_components = regular_components.copy()
    horizontal_lines.sort(key=lambda x: x['y'])
    
    i = 0
    while i < len(horizontal_lines) - 1:
        line1 = horizontal_lines[i]
        line2 = horizontal_lines[i + 1]
        
        vertical_gap = line2['y'] - (line1['y'] + line1['h'])
        x_overlap = min(line1['x'] + line1['w'], line2['x'] + line2['w']) - max(line1['x'], line2['x'])
        
        print(f"Checking lines gap={vertical_gap}, overlap={x_overlap}")
        
        if vertical_gap < 15 and x_overlap > 0:  # More lenient vertical gap
            x = min(line1['x'], line2['x'])
            y = line1['y']
            w = max(line1['x'] + line1['w'], line2['x'] + line2['w']) - x
            h = line2['y'] + line2['h'] - y
            
            print(f"Merging lines into equals sign: x={x}, y={y}, w={w}, h={h}")
            
            merged_components.append({
                'id': -1,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'is_equals': True
            })
            i += 2
        else:
            i += 1
    
    # Sort left to right
    merged_components.sort(key=lambda c: c['x'])
    
    # Draw final debug visualization
    debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, comp in enumerate(merged_components):
        x, y, w, h = comp['x'], comp['y'], comp['w'], comp['h']
        color = (0, 0, 255) if comp.get('is_equals', False) else (0, 255, 0)
        cv2.rectangle(debug_img, (x,y), (x+w,y+h), color, 1)
        cv2.putText(debug_img, f'{i}:{w}x{h}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite('debug_output/final_components.png', debug_img)
    
    # Extract symbols with proper padding and correct colors
    symbol_images = []
    for i, comp in enumerate(merged_components):
        x, y, w, h = comp['x'], comp['y'], comp['w'], comp['h']
        
        # Extract base symbol
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(gray.shape[1], x + w)
        y2 = min(gray.shape[0], y + h)
        symbol = thresh[y1:y2, x1:x2]
        
        # Invert the colors (255 - symbol)
        symbol = cv2.bitwise_not(symbol)
        
        # Calculate padding to make it square with plenty of border
        max_dim = max(w, h)
        border_size = int(max_dim * 0.5)  # 50% padding on each side
        
        # Add white padding (value=255 for white)
        padded = cv2.copyMakeBorder(
            symbol,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=255
        )
        
        # Resize to final size (28x28 for MNIST-style training data)
        final_symbol = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Save debug image
        cv2.imwrite(f'debug_output/symbol_{i}_x{x}_y{y}.png', final_symbol)
        symbol_images.append(final_symbol)
    
    return symbol_images

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
            symbol_images = segment_symbols(filepath)
            
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
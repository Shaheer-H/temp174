from flask import Flask, request, render_template, jsonify
import os
from model import DigitNet, OperatorNet
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.current_equation = []  # To store equation parts

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save and process the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            image_tensor = process_image(filepath)
            
            # Get predictions
            with torch.no_grad():
                # Try digit classification
                digit_output = digit_net(image_tensor)
                operator_output = operator_net(image_tensor)
                
                digit_probs = torch.nn.functional.softmax(digit_output, dim=1)
                operator_probs = torch.nn.functional.softmax(operator_output, dim=1)
                
                digit_conf, digit_pred = torch.max(digit_probs, 1)
                operator_conf, operator_pred = torch.max(operator_probs, 1)
                
                # Compare confidences to determine if it's a digit or operator
                if digit_conf > operator_conf:
                    detected_symbol = str(digit_pred.item())
                    confidence = float(digit_conf.item())
                else:
                    operator_idx = operator_pred.item()
                    detected_symbol = operator_mapping[operator_idx]
                    confidence = float(operator_conf.item())
            
            # Add symbol to current equation
            app.current_equation.append(detected_symbol)
            
            # Build the full equation string
            full_equation = ' '.join(app.current_equation)
            
            # Try to evaluate if it's a complete equation
            try:
                # Replace mathematical symbols with Python operators
                calc_eq = full_equation.replace('×', '*').replace('÷', '/')
                if '=' in calc_eq:
                    solution = "Equations not supported yet"
                else:
                    try:
                        result = eval(calc_eq)
                        solution = f"Result: {result}"
                    except:
                        solution = "Building equation: " + full_equation
            except:
                solution = "Building equation: " + full_equation
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'equation': full_equation,
                'solution': solution,
                'confidence': f"{confidence * 100:.2f}%",
                'last_detected': detected_symbol
            })
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return jsonify({'error': str(e)})
        
    return jsonify({'error': 'Invalid file type'})

# Add a new route to clear the equation
@app.route('/clear', methods=['POST'])
def clear_equation():
    app.current_equation = []
    return jsonify({'status': 'cleared'})

# Print debug information
print("Template folder:", app.template_folder)
print("Working directory:", os.getcwd())

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), UPLOAD_FOLDER)

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

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
                                        map_location=device))
    operator_net.load_state_dict(torch.load('weights/operator_net_best.pth', 
                                           map_location=device))
    print("Model weights loaded successfully")
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise

# Set models to evaluation mode
digit_net.eval()
operator_net.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(image_path)
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0).to(device)

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        return str(e), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            # Save and process the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            image_tensor = process_image(filepath)
            
            # Get predictions
            with torch.no_grad():
                # Try digit classification
                digit_output = digit_net(image_tensor)
                operator_output = operator_net(image_tensor)
                
                digit_probs = torch.nn.functional.softmax(digit_output, dim=1)
                operator_probs = torch.nn.functional.softmax(operator_output, dim=1)
                
                digit_conf, digit_pred = torch.max(digit_probs, 1)
                operator_conf, operator_pred = torch.max(operator_probs, 1)
                
                # Compare confidences to determine if it's a digit or operator
                if digit_conf > operator_conf:
                    detected_symbol = str(digit_pred.item())
                    confidence = float(digit_conf.item())
                else:
                    operator_idx = operator_pred.item()
                    detected_symbol = operator_mapping[operator_idx]
                    confidence = float(operator_conf.item())
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'equation': detected_symbol,
                'solution': "Not an equation yet. Upload more symbols to form an equation.",
                'confidence': f"{confidence * 100:.2f}%"
            })
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return jsonify({'error': str(e)})
        
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    print("Starting Flask app...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)
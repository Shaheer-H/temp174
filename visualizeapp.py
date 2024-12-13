from flask import Flask, request, render_template, jsonify
import os
import json
from model import CombinedNet, digit_classes, operator_classes
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch.nn.functional as F

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), UPLOAD_FOLDER
)

# Ensure required directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("debug_output", exist_ok=True)

# Determine device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load single combined model
model = CombinedNet().to(device)

# Load trained weights
try:
    model.load_state_dict(torch.load('weights/combined_net_best.pth', 
                                   map_location=device))
    print("Model weights loaded successfully")
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise

# Set model to evaluation mode
model.eval()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

#Ensure white background and black text
def segment_symbols(image_path):
    # Read and preprocess image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create a color image for visualization
    debug_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Get all components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # Draw all components with different colors
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [255, 255, 255]  # Background color
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        # Draw bounding box
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), colors[i].tolist(), 2)
        # Add component number
        cv2.putText(debug_img, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i].tolist(), 2)

    # Save initial components image
    cv2.imwrite("debug_output/all_components.png", debug_img)

    # Separate components into categories
    horizontal_lines = []
    regular_components = []
    min_area = 5

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue

        aspect_ratio = w / h if h > 0 else 0
        
        # Decimal point detection
        if (15 <= w <= 20 and 15 <= h <= 20 and 
            abs(w - h) <= 3 and area < 300):
            print(f"Potential decimal point found: area={area}, w={w}, h={h}")
            regular_components.append({
                "id": i, "x": x, "y": y, "w": w, "h": h, 
                "is_decimal": True
            })
        # Horizontal line detection
        elif aspect_ratio > 2 or (w > 15 and h <= 5):
            horizontal_lines.append({
                "id": i, "x": x, "y": y, "w": w, "h": h
            })
        else:   
            regular_components.append({
                "id": i, "x": x, "y": y, "w": w, "h": h
            })

    # Print initial components
    print("\nInitial components:")
    for i, comp in enumerate(regular_components):
        print(f"Regular {i}: x={comp['x']}, y={comp['y']}, w={comp['w']}, h={comp['h']}")
    
    print("\nHorizontal lines:")
    for i, line in enumerate(horizontal_lines):
        print(f"Line {i}: x={line['x']}, y={line['y']}, w={line['w']}, h={line['h']}")

    # Group horizontal lines by x-position
    right_lines = []  # Lines at x > 1700
    other_lines = []  # Other lines
    
    for line in horizontal_lines:
        if line["x"] > 1700:
            right_lines.append(line)
        else:
            other_lines.append(line)
    
    print("\nGrouped horizontal lines:")
    print("Right lines:", len(right_lines))
    for line in right_lines:
        print(f"  x={line['x']}, y={line['y']}, w={line['w']}, h={line['h']}")
    print("Other lines:", len(other_lines))
    for line in other_lines:
        print(f"  x={line['x']}, y={line['y']}, w={line['w']}, h={line['h']}")

    # Initialize merged_components with regular components
    merged_components = regular_components.copy()
    print("\nInitial merged components:")
    for i, comp in enumerate(merged_components):
        print(f"Component {i}: x={comp['x']}, y={comp['y']}, w={comp['w']}, h={comp['h']}")

    # Process equals sign (rightmost lines)
    if len(right_lines) >= 2:
        line1, line2 = right_lines[0], right_lines[1]
        vertical_gap = abs(line2["y"] - (line1["y"] + line1["h"]))
        width_diff = abs(line1["w"] - line2["w"])
        x_diff = abs(line1["x"] - line2["x"])
        
        print(f"\nChecking potential equals sign:")
        print(f"  Vertical gap: {vertical_gap}")
        print(f"  Width difference: {width_diff}")
        print(f"  X position difference: {x_diff}")
        
        if vertical_gap < 100 and width_diff < 50 and x_diff < 50:
            print(f"Found equals sign at y={line1['y']}")
            merged_components.append({
                "id": -1,
                "x": min(line1["x"], line2["x"]),
                "y": min(line1["y"], line2["y"]),
                "w": max(line1["w"], line2["w"]),
                "h": max(line2["y"] + line2["h"], line1["y"] + line1["h"]) - 
                     min(line1["y"], line2["y"]),
                "is_equals": True
            })

    # Process division (other lines) and remove associated decimal points
    if len(other_lines) >= 1:
        line = other_lines[0]
        # Find decimal points near the division line
        dots_near_line = []
        
        for comp in merged_components:
            if comp.get('is_decimal'):
                x_dist = abs(comp["x"] - line["x"])
                y_dist = abs(comp["y"] - line["y"])
                if y_dist < 100:  # Within 50 pixels vertically
                    dots_near_line.append(comp)
                    print(f"Found dot near line: x={comp['x']}, y={comp['y']}")
        
        if len(dots_near_line) >= 2:  # Need at least 2 dots for division
            print(f"Found division sign at y={line['y']} with {len(dots_near_line)} dots")
            
            # Create division component
            merged_components.append({
                "id": -1,
                "x": line["x"],
                "y": min(d["y"] for d in dots_near_line),
                "w": line["w"],
                "h": max(d["y"] + d["h"] for d in dots_near_line) - min(d["y"] for d in dots_near_line),
                "is_division": True
            })
            
            # Remove the decimal points that are part of division
            print(f"Removing {len(dots_near_line)} decimal points that are part of division")
            merged_components = [comp for comp in merged_components 
                               if comp not in dots_near_line]

    # Sort by x position (removed the equals sign filtering)
    merged_components.sort(key=lambda c: c["x"])
    
    # Create debug output for final components
    print("\nFinal components after division processing:")
    for i, comp in enumerate(merged_components):
        comp_type = 'regular'
        if comp.get('is_equals'): comp_type = 'equals'
        elif comp.get('is_division'): comp_type = 'division'
        elif comp.get('is_decimal'): comp_type = 'decimal'
        print(f"Component {i}: type={comp_type}, x={comp['x']}, y={comp['y']}")

    # Process components into symbols
    symbol_images = []
    print("\nProcessing components into symbols:")
    for i, comp in enumerate(merged_components):
        if comp.get('is_equals'):
            print(f"Component {i}: Adding equals sign")
            symbol_images.append('eq')
        elif comp.get('is_division'):
            print(f"Component {i}: Adding division sign")
            symbol_images.append('div')
        elif comp.get('is_decimal'):
            print(f"Component {i}: Adding decimal point")
            symbol_images.append('dec')
        else:
            print(f"Component {i}: Processing regular symbol at x={comp['x']}, y={comp['y']}")
            x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
            symbol_region = thresh[y:y+h, x:x+w]
            symbol_region = cv2.bitwise_not(symbol_region)
            processed_symbol = process_single_symbol(symbol_region, w, h)
            symbol_images.append(processed_symbol)

    return symbol_images

def process_single_symbol(symbol, x, y):
    """Process a single symbol image array"""
    #Our dataset is mostly comprised of 400x400 images
    target_size = 400
    
    # Calculate padding to make it square with border
    max_dim = max(x, y)
    border_size = int(max_dim * 0.5)  # 50% padding on each side
    
    # Add white padding
    padded = cv2.copyMakeBorder(
        symbol,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )
    
    # Resize to target size & Makes it less jagged
    processed_symbol = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    processed_symbol = cv2.GaussianBlur(processed_symbol, (3, 3), 0)

    # Thresholding to clean up any artifacts
    _, processed_symbol = cv2.threshold(processed_symbol, 127, 255, cv2.THRESH_BINARY)

    return processed_symbol

def to_tensor(image_array, transform, device):
    """Convert image array to tensor with debugging"""
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    
    # Save the image before transforms
    image.save(f"debug_output/before_transform.png")
    
    # Apply each transform separately for debugging
    resized = transforms.Resize((32, 32))(image)
    resized.save(f"debug_output/after_resize.png")
    
    # Convert to tensor
    tensor = transforms.ToTensor()(resized)
    
    # Convert tensor back to image for visualization
    pil_image = transforms.ToPILImage()(tensor)
    pil_image.save(f"debug_output/after_to_tensor.png")
    
    # Apply normalization
    normalized = transforms.Normalize((0.5,), (0.5,))(tensor)
    
    # Convert normalized tensor to image for visualization
    # Denormalize first: pixel = (pixel * std) + mean
    denorm = normalized.clone()
    denorm = denorm * 0.5 + 0.5
    denorm_image = transforms.ToPILImage()(denorm)
    denorm_image.save(f"debug_output/after_normalize.png")
    
    # Add batch dimension and move to device
    final_tensor = normalized.unsqueeze(0).to(device)
    
    return final_tensor

# Define transform globally
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_prediction(image_tensor, model, symbol_num=0, type_threshold=0.8):
    """Get prediction using the three-headed model with threshold-based decision"""
    with torch.no_grad():
        # Use the new predict_with_threshold method
        pred_type, pred_idx, confidence = model.predict_with_threshold(image_tensor, type_threshold)
        
        # Print detailed debugging information
        print(f"\nDetailed prediction for symbol {symbol_num}:")
        type_out, digit_out, operator_out = model(image_tensor)
        
        # Get all probabilities for debugging
        type_probs = F.softmax(type_out, dim=1)
        digit_probs = F.softmax(digit_out, dim=1)
        operator_probs = F.softmax(operator_out, dim=1)
        
        print(f"Type prediction confidence: {type_probs.max().item():.4f}")
        
        print("\nTop 3 Digit probabilities:")
        digit_probs_np = digit_probs.cpu().numpy()[0]
        top_digits = np.argsort(digit_probs_np)[-3:][::-1]
        for idx in top_digits:
            print(f"{digit_classes[idx]}: {digit_probs_np[idx]:.4f}")
            
        print("\nTop 3 Operator probabilities:")
        operator_probs_np = operator_probs.cpu().numpy()[0]
        top_operators = np.argsort(operator_probs_np)[-3:][::-1]
        for idx in top_operators:
            print(f"{operator_classes[idx]}: {operator_probs_np[idx]:.4f}")
        
        # Get final symbol based on prediction type
        if pred_type == 'digit':
            symbol = digit_classes[pred_idx]
            print(f"\nFinal prediction: DIGIT {symbol} (confidence: {confidence:.4f})")
        else:
            symbol = operator_classes[pred_idx]
            print(f"\nFinal prediction: OPERATOR {symbol} (confidence: {confidence:.4f})")
        
        # Save debug information
        save_predictions(
            digit_probs.cpu().numpy(),
            operator_probs.cpu().numpy(),
            pred_idx if pred_type == 'digit' else -1,
            confidence if pred_type == 'digit' else 0.0,
            pred_idx if pred_type == 'operator' else -1,
            confidence if pred_type == 'operator' else 0.0,
            symbol_num
        )
            
        return symbol, confidence

def save_predictions(digit_probs, oper_probs, digi_pred, digit_conf, oper_pred, oper_conf, symbol_num):
    #Format probabilities for verification
    modelPredictions = {
        'digits_probabilities': {
            digit_classes[i]: float(prob) for i, prob in enumerate(digit_probs[0])
        },
        'operator_probabilities': {
            operator_classes[i]: float(prob) for i, prob in enumerate(oper_probs[0])
        },
        'most_likely_digit': {
            'prediction': digit_classes[digi_pred] if digi_pred != -1 else "none",
            'confidence': digit_conf
        },
        'most_likely_operator': {
            'prediction': operator_classes[oper_pred] if oper_pred != -1 else "none",
            'confidence': oper_conf
        }
    }    

    #Format probabilities for verification
    debug_file = f'debug_output/prediction_debug_{symbol_num}.json'
    with open(debug_file, 'w') as f:
        json.dump(modelPredictions, f, indent=2)

def clear_debug_output():
    """Clear all files in debug_output directory"""
    for file in os.listdir("debug_output"):
        file_path = os.path.join("debug_output", file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    # Clear debug output directory
    clear_debug_output()
    
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
            print(f"Found {len(symbol_images)} symbols")
            
            if not symbol_images:
                return jsonify({'error': 'No symbols detected in image'})
            
            # Process each symbol
            detected_symbols = []
            confidences = []
            
            for i, symbol_image in enumerate(symbol_images):
                # Handle special symbols
                if isinstance(symbol_image, str):
                    if symbol_image == 'dec':
                        detected_symbols.append('.')
                    elif symbol_image == 'eq':
                        detected_symbols.append('=')
                    elif symbol_image == 'sub':
                        detected_symbols.append('-')
                    elif symbol_image == 'div':
                        detected_symbols.append('÷')
                    confidences.append(1.0)
                    continue
                
                # Normal processing for other symbols
                image_tensor = to_tensor(symbol_image, transform, device)
                symbol, confidence = get_prediction(image_tensor, model, symbol_num=i)
                print(f"Symbol {i}: {symbol} (confidence: {confidence:.2f})")
                detected_symbols.append(symbol)
                confidences.append(confidence)
            
            # Build equation string
            equation = ' '.join(detected_symbols)
            
            # Try to evaluate the equation
            try:
                if '=' in equation:
                    # Split equation at equals sign and evaluate left side
                    left_side = equation.split('=')[0]
                    # Convert operators before evaluation
                    left_side = left_side.strip().replace('×', '*').replace('÷', '/').replace('add', '+').replace('sub', '-').replace('mul', '*')
                    
                    # Remove spaces between consecutive digits
                    parts = left_side.split()
                    cleaned_expr = ''
                    current_number = ''
                    
                    for part in parts:
                        if part.isdigit():
                            current_number += part
                        else:
                            if current_number:
                                cleaned_expr += ' ' + current_number + ' '
                                current_number = ''
                            cleaned_expr += part + ' '
                    
                    # Add any remaining number
                    if current_number:
                        cleaned_expr += current_number
                    
                    cleaned_expr = cleaned_expr.strip()
                    print(f"Cleaned expression: {cleaned_expr}")  # Debug print
                    
                    try:
                        result = eval(cleaned_expr)
                        solution = f"Result: {result}"
                    except Exception as calc_error:
                        print(f"Calculation error: {calc_error}")
                        solution = equation
                else:
                    # Handle normal expressions (same cleaning for non-equation expressions)
                    calc_eq = equation.replace('×', '*').replace('÷', '/').replace('add', '+').replace('sub', '-')
                    parts = calc_eq.split()
                    cleaned_expr = ''
                    for i, part in enumerate(parts):
                        if part.isdigit() or part == '.':
                            if i > 0 and (parts[i-1].isdigit() or parts[i-1] == '.'):
                                cleaned_expr += part
                            else:
                                cleaned_expr += ' ' + part
                        else:
                            cleaned_expr += ' ' + part
                    
                    result = eval(cleaned_expr.strip())
                    solution = f"Result: {result}"
            except Exception as e:
                print(f"Evaluation error: {e}")
                solution = equation
            
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


if __name__ == "__main__":
    print("Starting Flask app...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)

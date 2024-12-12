from flask import Flask, request, render_template, jsonify
import os
import json
from model import CombinedNet
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

# Determine device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load single model instead of two
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

    # Get all components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8
    )

    # Debug: Print all component stats
    print("\nAll Components:")
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect = w / h if h > 0 else 0
        print(
            f"Component {i}: x={x}, y={y}, w={w}, h={h}, area={area}, aspect={aspect:.2f}"
        )

    # Separate horizontal lines and regular components
    horizontal_lines = []
    regular_components = []
    min_area = 5

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue

        aspect_ratio = w / h if h > 0 else 0
        if (aspect_ratio > 2 and h <= 15) or (w > 15 and h <= 5):
            print(
                f"Found horizontal line: Component {i} with aspect ratio {aspect_ratio:.2f}"
            )
            horizontal_lines.append({"id": i, "x": x, "y": y, "w": w, "h": h})
        else:   
            regular_components.append({"id": i, "x": x, "y": y, "w": w, "h": h})

    # Draw debug visualization for pre-merge components
    debug_pre = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for comp in horizontal_lines:
        x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
        cv2.rectangle(debug_pre, (x, y), (x + w, y + h), (0, 0, 255), 1)
    for comp in regular_components:
        x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
        cv2.rectangle(debug_pre, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imwrite("debug_output/pre_merge.png", debug_pre)

    # Merge horizontal lines into equals signs
    merged_components = regular_components.copy()
    horizontal_lines.sort(key=lambda x: x["y"])

    i = 0
    while i < len(horizontal_lines) - 1:
        line1 = horizontal_lines[i]
        line2 = horizontal_lines[i + 1]

        vertical_gap = line2["y"] - (line1["y"] + line1["h"])
        x_overlap = min(line1["x"] + line1["w"], line2["x"] + line2["w"]) - max(
            line1["x"], line2["x"]
        )

        if vertical_gap < 15 and x_overlap > 0:
            x = min(line1["x"], line2["x"])
            y = line1["y"]
            w = max(line1["x"] + line1["w"], line2["x"] + line2["w"]) - x
            h = line2["y"] + line2["h"] - y

            merged_components.append(
                {"id": -1, "x": x, "y": y, "w": w, "h": h, "is_equals": True}
            )
            i += 2
        else:
            i += 1

    # Sort components in natural reading order
    row_threshold = max(comp["h"] for comp in merged_components) * 0.5
    merged_components.sort(key=lambda c: c["y"])

    rows = []
    current_row = [merged_components[0]]

    for comp in merged_components[1:]:
        if abs(comp["y"] - current_row[0]["y"]) < row_threshold:
            current_row.append(comp)
        else:
            rows.append(sorted(current_row, key=lambda c: c["x"]))
            current_row = [comp]

    if current_row:
        rows.append(sorted(current_row, key=lambda c: c["x"]))

    merged_components = [comp for row in rows for comp in row]

    # Draw final debug visualization
    debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i, comp in enumerate(merged_components):
        x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
        color = (0, 0, 255) if comp.get("is_equals", False) else (0, 255, 0)
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 1)
        cv2.putText(
            debug_img,
            f"{i}:{w}x{h}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )
    cv2.imwrite("debug_output/final_components.png", debug_img)

    # Extract and process symbols
    symbol_images = []
    for i, comp in enumerate(merged_components):
        x, y, w, h = comp["x"], comp["y"], comp["w"], comp["h"]
        
        # Extract region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(gray.shape[1], x + w)
        y2 = min(gray.shape[0], y + h)

        symbol_region = thresh[y1:y2, x1:x2]
        symbol_region = cv2.bitwise_not(symbol_region)

        #Save preprocessed image
        cv2.imwrite(f"debug_output/preprocessed_symbol_{i}_x{x}_y{y}.png", symbol_region)

        # Process symbol
        processed_symbol = process_single_symbol(symbol_region, (x2 - x1), (y2 - y1))
        
        # Save debug image
        cv2.imwrite(f"debug_output/processed_symbol_{i}_x{x}_y{y}.png", processed_symbol)
        symbol_images.append(processed_symbol)

    return symbol_images

#Move all the above processing into the following function to reduce clutter
def process_single_symbol(symbol, x, y):
    """Process a single symbol image array"""

    #Our dataset is mostly comprised of 400x400 images
    target_size = 400
    
    # Calculate padding to make it square with border
    max_dim = max(x, y)
    border_size = int(max_dim * 0.5)  # 50% padding on each side
    
    #Maybe should try centering the image

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

    #Reduce noise
    kernel = np.ones((2,2), np.uint8)
    processed_symbol = cv2.adaptiveThreshold(processed_symbol, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return processed_symbol

def to_tensor(image_array, transform, device):
     # Convert numpy array to PIL Image
    image = Image.fromarray(image_array)
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0).to(device)

    return image_tensor

def get_prediction(image_tensor, model):
    """Get prediction using the three-headed model"""
    with torch.no_grad():
        type_out, digit_out, operator_out = model(image_tensor)
        
        # Get type prediction first
        type_probs = F.softmax(type_out, dim=1)
        type_conf, type_pred = torch.max(type_probs, 1)
        
        # Based on type, get either digit or operator prediction
        if type_pred.item() == 0:  # Digit
            digit_probs = F.softmax(digit_out, dim=1)
            conf, pred = torch.max(digit_probs, 1)
            symbol = str(pred.item())  # Convert digit to string
        else:  # Operator
            operator_probs = F.softmax(operator_out, dim=1)
            conf, pred = torch.max(operator_probs, 1)
            from model import operator_classes  # Import operator_classes
            symbol = operator_classes[pred.item()]
            
        return symbol, conf.item()

def save_predictions(digit_probs, oper_probs, digi_pred, digit_conf, oper_pred, oper_conf, symbol_num):
    from model import operator_classes  # Import operator_classes
    
    #Format probabilities for verification
    modelPredictions = {
        'digits_probabilities' : {
            str(i): float(prob) for i, prob in enumerate(digit_probs[0])
        },
        'operator_probabilities': {
            operator_classes[i]: float(prob) for i, prob in enumerate(oper_probs[0])
        },
        'most_likely_digit': {
            'prediction': str(digi_pred),
            'confidence': digit_conf
        },
        'most_likely_operator': {
            'prediction': operator_classes[oper_pred],
            'confidence': oper_conf
        }
    }    

    debug_file = f'debug_output/prediction_debug_{symbol_num}.json'

    #Write file with model predictions in debug_outputs
    with open(debug_file, 'w') as f:
        json.dump(modelPredictions, f, indent=2)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Segment the image into individual symbols
            symbol_images = segment_symbols(filepath)

            if not symbol_images:
                return jsonify({"error": "No symbols detected in image"})

            # Process each symbol
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

            detected_symbols = []
            confidences = []

            symbol_count = 0
            for symbol_image in symbol_images:
                # Convert to uint8 if not already
                if symbol_image.dtype != np.uint8:
                    symbol_image = (symbol_image * 255).astype(np.uint8)
                image_tensor = to_tensor(symbol_image, transform, device)
                symbol, confidence = get_prediction(
                    image_tensor, model
                )
                symbol_count += 1
                detected_symbols.append(symbol)
                confidences.append(confidence)

            # Build equation string
            equation = "".join(detected_symbols)

            # Try to evaluate the equation
            try:
                calc_eq = equation.replace("×", "*").replace("÷", "/")
                if "=" in calc_eq:
                    expression = equation.replace("=","")
                    operation = "simplify"
                    req = requests.get(f"https://newton.now.sh/api/v2/{operation}/{expression}")
                    data = req.json()
                    solution = data["result"]
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

            return jsonify(
                {
                    "equation": equation,
                    "solution": solution,
                    "confidence": f"{avg_confidence * 100:.2f}%",
                    "num_symbols": len(detected_symbols),
                }
            )

        except Exception as e:
            print(f"Error in prediction: {e}")
            return jsonify({"error": str(e)})

    return jsonify({"error": "Invalid file type"})


if __name__ == "__main__":
    print("Starting Flask app...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)

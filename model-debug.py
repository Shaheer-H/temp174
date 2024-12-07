import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from model import DigitNet, OperatorNet

def get_latest_upload():
    """Get the most recent file from the uploads directory."""
    upload_dir = 'uploads'
    if not os.path.exists(upload_dir):
        print(f"Upload directory '{upload_dir}' does not exist!")
        return None
    
    files = [f for f in os.listdir(upload_dir) if os.path.isfile(os.path.join(upload_dir, f))]
    if not files:
        print("No files found in uploads directory!")
        return None
    
    latest_file = max([os.path.join(upload_dir, f) for f in files], 
                     key=os.path.getmtime)
    return latest_file

def debug_model_prediction(image_path):
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    digit_net = DigitNet().to(device)
    operator_net = OperatorNet().to(device)
    
    # Load weights
    try:
        digit_net.load_state_dict(torch.load('weights/digit_net_best.pth', 
                                           map_location=device,
                                           weights_only=True))
        operator_net.load_state_dict(torch.load('weights/operator_net_best.pth', 
                                              map_location=device,
                                              weights_only=True))
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    # Set to eval mode
    digit_net.eval()
    operator_net.eval()
    
    # Load and preprocess image
    print(f"Loading image from: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist!")
        return
        
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Failed to load image {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Binary threshold
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Display images
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(132)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    plt.subplot(133)
    plt.imshow(labels, cmap='nipy_spectral')
    plt.title(f'Connected Components ({num_labels-1} symbols)')
    plt.show()
    
    # Process each symbol
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Debug each symbol
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 20:  # Skip noise
            continue
            
        # Extract symbol
        symbol = binary[y:y+h, x:x+w]
        resized = cv2.resize(symbol, (28, 28))
        
        # Display symbol
        plt.figure(figsize=(5, 5))
        plt.imshow(resized, cmap='gray')
        plt.title(f'Symbol {i}')
        plt.show()
        
        # Process for network
        pil_image = Image.fromarray(resized)
        image_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            digit_output = digit_net(image_tensor)
            operator_output = operator_net(image_tensor)
            
            # Get probabilities
            digit_probs = torch.nn.functional.softmax(digit_output, dim=1)
            operator_probs = torch.nn.functional.softmax(operator_output, dim=1)
            
            # Print raw outputs
            print(f"\nSymbol {i} Raw Outputs:")
            print("Digit logits:", digit_output.squeeze().cpu().numpy())
            print("Operator logits:", operator_output.squeeze().cpu().numpy())
            
            # Get predictions
            digit_conf, digit_pred = torch.max(digit_probs, 1)
            operator_conf, operator_pred = torch.max(operator_probs, 1)
            
            print(f"\nPredictions for symbol at ({x}, {y}):")
            print(f"Digit prediction: {digit_pred.item()} (confidence: {digit_conf.item()*100:.2f}%)")
            
            operator_mapping = {0: '+', 1: '.', 2: '÷', 3: '=', 4: '×', 5: '-', 6: 'x', 7: 'y', 8: 'z'}
            print(f"Operator prediction: {operator_mapping[operator_pred.item()]} (confidence: {operator_conf.item()*100:.2f}%)")

if __name__ == "__main__":
    latest_upload = get_latest_upload()
    if latest_upload:
        print(f"Processing latest upload: {latest_upload}")
        debug_model_prediction(latest_upload)
    else:
        print("No image found to process!")

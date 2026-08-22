import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Loader for Return Risk Model
def load_return_risk_model():
    model_path = "return_risk_model.pkl"
    if not os.path.exists(model_path):
        model_path = "/workspace/artifacts/return_risk_model.pkl"
    if not os.path.exists(model_path):
        model_path = "../models/return_risk_model.pkl"
    
    # Use joblib to load scikit-learn pipeline
    model = joblib.load(model_path)
    return model

# 2. Loader for Product Image Classifier
def load_image_classifier():
    model_path = "product_classifier.pt"
    if not os.path.exists(model_path):
        model_path = "/workspace/artifacts/product_classifier.pt"
    if not os.path.exists(model_path):
        model_path = "models/product_classifier.pt"
    
    # Simple architecture load to match what was trained
    model = models.resnet18(weights=None)
    model.fc = nn.Identity() # Freeze backbone as in training
    
    class ClassifierHead(nn.Module):
        def __init__(self, input_dim=512, hidden_dim=128, num_classes=10):
            super().__init__()
            # Must match name "net" from training
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )
        def forward(self, x):
            return self.net(x)
            
    class CombinedModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)
            
    head = ClassifierHead()
    combined_model = CombinedModel(model, head)
    
    # Load state dict
    combined_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    combined_model.eval()
    return combined_model

# 3. Tool: check_return_risk
def check_return_risk(order_details: dict) -> dict:
    """
    Predicts the return risk of an e-commerce order using the saved Random Forest model.
    Maps prediction probabilities to "Low", "Medium", and "High" risk buckets using t*_rf = 0.46.
    """
    model = load_return_risk_model()
    
    # Construct a DataFrame from details
    df_input = pd.DataFrame([order_details])
    
    # Predict probability of class 1 (Returned)
    prob = model.predict_proba(df_input)[0, 1]
    
    # Map to risk buckets
    threshold = 0.46
    if prob < threshold:
        risk_bucket = "Low"
    elif prob < (threshold + 0.15): # 0.46 to 0.61
        risk_bucket = "Medium"
    else:
        risk_bucket = "High"
        
    return {
        "return_probability": float(prob),
        "risk_bucket": risk_bucket,
        "recommendation": "Approve standard processing." if risk_bucket == "Low" else "Schedule verification callback." if risk_bucket == "Medium" else "Flag for customer-relations review before dispatch."
    }

# 4. Tool: classify_product_image
def classify_product_image(image_path: str) -> dict:
    """
    Loads your trained PyTorch image classifier and identifies the category of a sample image.
    """
    # Verify image exists
    if not os.path.exists(image_path):
        image_path = os.path.join("/workspace/artifacts", os.path.basename(image_path))
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    # Categories definition
    categories = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    model = load_image_classifier()
    
    # Define preprocessing pipeline matching training
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert("L") # load as grayscale as Fashion-MNIST is L
    img_t = preprocess(img).unsqueeze(0) # add batch dim
    
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1).flatten()
        max_idx = torch.argmax(probs).item()
        
    return {
        "predicted_category": categories[max_idx],
        "confidence": float(probs[max_idx]),
        "all_probabilities": {categories[i]: float(probs[i]) for i in range(len(categories))}
    }

# Test block
if __name__ == "__main__":
    # Test order risk details
    sample_order = {
        "product_category": "Apparel",
        "price_inr": 1500.0,
        "discount_pct": 25.0,
        "payment_method": "COD",
        "customer_tenure_days": 180,
        "num_previous_orders": 4,
        "num_previous_returns": 2,
        "delivery_distance_km": 150.0,
        "delivery_days": 3,
        "is_weekend_order": 1,
        "rating_given": np.nan
    }
    
    print("Testing check_return_risk...")
    res_risk = check_return_risk(sample_order)
    print("Risk Outcome:", res_risk)
    
    print("\nTesting classify_product_image...")
    sample_img_path = "/workspace/artifacts/07_sneaker.png"
    if os.path.exists(sample_img_path):
        res_img = classify_product_image(sample_img_path)
        print("Image Classification Outcome:", res_img["predicted_category"], "| Confidence:", round(res_img["confidence"], 4))
    else:
        print("Sample image not found for test.")

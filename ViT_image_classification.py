from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch

# Define the image path
image_path = "./image/lamborghini.png"  # Specify the path to your robot image

# Load the ViT model and image processor
model_name = "google/vit-base-patch16-224"  # Use the ViT model of your choice
model = ViTForImageClassification.from_pretrained(model_name)
image_processor = ViTImageProcessor.from_pretrained(model_name)

# Load an image
image = Image.open(image_path)

# Preprocess the image using the image processor
inputs = image_processor(images=image, return_tensors="pt")

# Make predictions
outputs = model(**inputs)
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1)

# Get class labels (optional)
class_labels = model.config.id2label

# Print the predicted class label
print("Predicted Class:", class_labels[predicted_class.item()])

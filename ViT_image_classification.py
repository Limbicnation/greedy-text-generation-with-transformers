from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from transformers import pipeline

# Load the image
image_path = "image/robot-1.jpg"  # Replace with the path to your image
img = Image.open(image_path)

# Display the image
plt.imshow(img)
plt.axis("off")
plt.show()

# Define the model to use
model_name = "google/vit-base-patch16-224"  # Replace with the model you want to use

# Create an image classification pipeline
image_classifier = pipeline("image-classification", model=model_name)

# Image classification
preds = image_classifier(image_path)
preds_df = pd.DataFrame(preds)
print(preds_df)

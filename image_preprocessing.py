from PIL import Image
import numpy as np

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized)
    image_normalized = image_array / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    return image_batch

# Test the function
preprocessed_image = preprocess_image("path_to_image.jpg")
print(preprocessed_image.shape)  # Output: (1, 224, 224, 3)

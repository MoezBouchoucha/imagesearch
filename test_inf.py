import os 
from imagesearch import ImageSearchEngine
import tensorflow as tf 
from transformers import ViTImageProcessor, TFViTModel
from PIL import Image  

# Path to your image
image_path = r"test\images.jpg"  # Replace with your image path


feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = TFViTModel.from_pretrained('google/vit-base-patch16-224')

def featuregetter(image_path,feature_extractor=feature_extractor,model=model):
    # Load the image using PIL
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format

    inputs = feature_extractor(images=image, return_tensors="tf")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state[:,0,:]
    normalized_features = tf.linalg.l2_normalize(last_hidden_states, axis=-1)
    return normalized_features

# print("Predicted class:", last_hidden_states.shape)
print(featuregetter(image_path).shape[1])

ise=ImageSearchEngine()
print(1)
l=[os.path.join("data",i) for i in os.listdir("data")]
print(l)
print(2)
ise.encoder=featuregetter
print(2.5)
ise.set_index_from_examples(l)
print(3)
ise.train_index(l)
print(4)
ise.add_items(l,l)
print(5)
ise.search(r"test\images.jpg",3,10)
print(6)
ise.save("image_db")
print(7)
ise.load("image_db",featuregetter)
print(8)
ise.search(r"test\images.jpg",3,10)
print("done sucessfuly")
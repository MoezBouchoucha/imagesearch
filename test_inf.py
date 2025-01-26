import os 
from imagesearch import ImageSearchEngine
import tensorflow as tf 
from transformers import ViTImageProcessor, TFViTModel
from PIL import Image  

import streamlit as st
import tempfile

if __name__=="__main__":
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
    # print(featuregetter(image_path).shape[1])

    ise=ImageSearchEngine()
    # print(1)
    # l=[os.path.join("data",i) for i in os.listdir("data")]
    # print(l)
    # print(2)
    # ise.encoder=featuregetter
    # print(2.5)
    # ise.set_index_from_examples(l)
    # print(3)
    # ise.train_index(l)
    # print(4)
    # ise.add_items(l,l)
    # print(5)
    # ise.search(r"test\images.jpg",3,10)
    # print(6)
    # ise.save("image_db")
    # print(7)
    ise.load("image_db",featuregetter)
    # print(8)
    # ise.search(r"test\images.jpg",3,10)
    # print("done sucessfuly")







    # Import your FAISS image search class here
    # from your_module import FaissImageSearch

    st.set_page_config(page_title="Image Search Engine", layout="wide")

    # # Initialize your FAISS system
    # # Make sure to update the index path according to your setup
    # @st.cache_resource
    # def load_faiss_system():
    #     return FaissImageSearch(index_path="path/to/your/faiss.index", 
    #                            image_paths="path/to/your/image_dataset.csv")

    # faiss_system = load_faiss_system()

    st.title("🔍 Image Search Engine")
    st.markdown("Drag and drop an image to find similar images")

    # File uploader section
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Drag and drop an image file here"
    )

    col1, col2 = st.columns([1, 3])

    if uploaded_file is not None:
        # Display uploaded image
        with col1:
            st.subheader("Your Image")
            uploaded_image = Image.open(uploaded_file)
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
        # Process the image and find similar images
        with col2:
            st.subheader("Similar Images")
            
            with st.spinner("Searching for similar images..."):
                # Save uploaded file to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    uploaded_image.save(tmp_file.name, format="JPEG")
                    
                    # Use your FAISS system to find similar images
                    try:
                        d,I = ise.search(
                            tmp_file.name, 
                            2,
                            verbose=False
                        )
                        similar_images = [ise.metadata[i] for i in I[0]]
                        print(similar_images)
                    except Exception as e:
                        st.error(f"Error during search: {str(e)}")
                        similar_images = []
                
                # Display similar images in a grid
                if similar_images:
                    cols = st.columns(4)  # Adjust number of columns as needed
                    for idx, img_path in enumerate(similar_images):
                        if os.path.exists(img_path):
                            try:
                                img = Image.open(img_path)
                                with cols[idx % 4]:  # This creates a 4-column grid
                                    st.image(
                                        img,
                                        caption=os.path.basename(img_path),
                                        use_container_width=True
                                    )
                            except Exception as e:
                                st.error(f"Error loading image {img_path}: {str(e)}")
                        else:
                            st.error(f"Image not found: {img_path}")
                else:
                    st.warning("No similar images found")
    else:
        col1.info("Please upload an image to search")
        col2.image("https://www.pbs.org/wnet/nature/files/2020/06/black-white-and-yellow-tiger-sitting-on-a-beige-sand-during-47312-scaled.jpg", 
                use_container_width=True)

    # Add some style
    st.markdown("""
    <style>
        .stImage img {
            border-radius: 10px;
            transition: transform 0.3s;
        }
        .stImage img:hover {
            transform: scale(1.05);
        }
        [data-testid="stFileUploader"] {
            border: 2px dashed #4CAF50;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
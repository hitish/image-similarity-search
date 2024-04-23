import streamlit as st
import numpy as np
import faiss
import torch
from torchvision import transforms
from torchvision.models import vgg16
from PIL import Image
import os
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Load embeddings and index (if saved from image_embedding.py)
#embeddings = np.load("embeddings.npy")
faiss_index = faiss.read_index("faiss_index.bin")

with open("img_dict.pkl", "rb") as f:
    img_dict = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),  # Resize image to 256x256
    transforms.CenterCrop(224),  # Crop central 224x224 region
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize based on ImageNet statistics
])

# Load pre-trained VGG16 model (without final classification layer)
model = vgg16(pretrained=True).features.to(device)

def get_image_embedding(query_img):
  img = transform(query_img)  # Load and transform image
  img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
  with torch.no_grad():  # Disable gradient calculation for efficiency
    embedding = model(img)  # Pass image through VGG16
  embedding = embedding.flatten(start_dim=0)  # Flatten feature map to vector
  return embedding.cpu().detach().numpy()  # Move embedding to CPU and numpy array


def search_similar(query_img):
  # Preprocess query image
  query_embedding = get_image_embedding(query_img)

  # Search for nearest neighbors using Faiss
  d, indices = faiss_index.search(query_embedding.reshape(1, -1), k=5)  # Search for 5 neighbors

  # Show results (replace with your image display logic)
  st.subheader("Similar Images:")
  for i, idx in enumerate(indices[0]):
    st.image(img_dict[idx], width=200)
    st.write(f"Distance: {d[0][i]}")

st.title("Image Similarity Search")
st.subheader("Demo app by Hitish Singla")
uploaded_file = st.file_uploader("Choose an image of Bear,Bison,Cat,Catterpillar,Chimpanzee,Cow,Cockroach,Dog,Dolphin,Eagle,Elephant,Goat,Leopard,Mosquito,Octopus", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  query_img = Image.open(uploaded_file)
  search_similar(query_img)


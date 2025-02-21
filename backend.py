
import os
import torch
import clip
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, Query
from typing import List
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# Initialize FastAPI app
app = FastAPI()

# Load CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load images and extract features
image_folder = "C:/Users/Khuth/OneDrive/Desktop/Koketso Malope/Multi-Model Retrieval/test_data_v2"
image_files = sorted(os.listdir(image_folder))[:500]  # Use only 500 images

image_embeddings = []
image_paths = []

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_embeddings.append(image_features.cpu().numpy())
    image_paths.append(image_path)

image_embeddings = np.vstack(image_embeddings)

# Apply PCA to reduce image embeddings to 384 dimensions
pca = PCA(n_components=384)
image_embeddings_pca = pca.fit_transform(image_embeddings)  # Apply PCA to images

# Fit Nearest Neighbors model using PCA-reduced image embeddings
knn = NearestNeighbors(n_neighbors=5, metric="cosine")
knn.fit(image_embeddings_pca)  # Use the transformed image embeddings

@app.get("/search")
def search_images(query: str = Query(..., description="Text query to search for images")):
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([query]).to(device)).cpu().numpy()
    
    # Apply PCA to reduce text embeddings to 384 dimensions
    text_features_pca = pca.transform(text_features)  # Apply same PCA transformation

    distances, indices = knn.kneighbors(text_features_pca)
    results = [image_paths[i] for i in indices[0]]
    return {"query": query, "results": results}



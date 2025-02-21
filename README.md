# Multi-Modal-Image-Retrieval-System
This project implements a multi-modal image retrieval system that allows users to search for images using text descriptions. It features a user-friendly front-end interface and utilizes open-source models and libraries to return the top-K matching images from a provided dataset

This repository contains a Multi-Modal Image Retrieval System that allows users 
to search for images using text queries. The system uses OpenAI's CLIP (ViT-B/32) model 
to generate image and text embeddings, applies Principal Component Analysis (PCA) for 
dimensionality reduction, and utilizes K-Nearest Neighbors (KNN) for similarity search. 
The backend is implemented using FastAPI, and the frontend is built with Streamlit.

## Table of Contents
- [Setup and Installation](#setup-and-installation)
- [Running the System](#running-the-system)
- [Testing the System](#testing-the-system)
- [Assumptions Made](#assumptions-made)
- [Next Steps & Improvements](#next-steps--improvements)

---

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/multi-modal-retrieval.git
cd multi-modal-retrieval
```

### 2. Create a Virtual Environment and Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate  # For Windows
pip install -r requirements.txt
```

### 3. Install PyTorch and CLIP
Ensure you have **PyTorch** installed based on your system configuration.  
Check [PyTorch installation guide](https://pytorch.org/get-started/locally/).  
For example, to install PyTorch with CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Then install **CLIP**:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### 4. Install FastAPI and Streamlit
```bash
pip install fastapi uvicorn streamlit scikit-learn pillow requests
```

---

## Running the System

### Step 1: Start the FastAPI Backend
```bash
uvicorn main:app --reload
```
- This will start the FastAPI server at `http://127.0.0.1:8000/`.
- You can access the API documentation at `http://127.0.0.1:8000/docs`.

### Step 2: Start the Streamlit Frontend
```bash
streamlit run app.py
```
- This will open the Streamlit web interface where users can input text queries and retrieve relevant images.

---

## Testing the System

### 1. Testing the FastAPI Endpoint Manually
You can test the API using **cURL** or **Postman**:
```bash
curl -X 'GET' 'http://127.0.0.1:8000/search?query=cat' -H 'accept: application/json'
```

### 2. Running Automated Tests
To ensure everything is working correctly, run the test script:
```bash
pytest tests/
```
Ensure you have **pytest** installed:
```bash
pip install pytest
```

---

## Assumptions Made

1. **Dataset Availability**  
   - The system assumes that a folder `test_data_v2` exists at `C:/Users/Khuth/OneDrive/Desktop/Koketso Malope/Multi-Model Retrieval/`, containing images.  
   - If using a different folder, update the `image_folder` path in the script.

2. **CLIP Model**  
   - The system uses **CLIP (ViT-B/32)** model, assuming it is suitable for multi-modal retrieval.

3. **Dimensionality Reduction**  
   - PCA reduces embeddings from CLIP to **384 dimensions** for efficient similarity search.

4. **Nearest Neighbors Search**  
   - The system uses **K-Nearest Neighbors (KNN) with cosine similarity** for retrieval.

---

## Next Steps & Improvements

- Optimize **image embedding storage** to avoid recomputation on each run.  
- Implement **caching** to speed up search queries.  
- Extend to support **larger datasets and indexing methods (e.g., FAISS)**.  
- Improve **UI/UX** for better search experience.  

---

This README provides all necessary steps to set up, run, and test the Multi-Modal Image Retrieval System.

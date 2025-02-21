
import streamlit as st
import requests
import os
from typing import List

st.title("Multi-Modal Image Retrieval")
st.write("Enter a text query to search for relevant images.")

query = st.text_input("Search Query")
if st.button("Search"):
    if query:
        response = requests.get("http://127.0.0.1:8000/search", params={"query": query})
        if response.status_code == 200:
            data = response.json()
            st.write(f"Results for: {data['query']}")
            for img_path in data["results"]:
                st.image(img_path, caption=os.path.basename(img_path))
        else:
            st.error("Error retrieving results. Make sure the FastAPI backend is running.")


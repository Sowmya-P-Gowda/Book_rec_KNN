import streamlit as st
import pickle
import numpy as np

# Load necessary files
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('book_user_matrix.pkl', 'rb') as f:
    matrix = pickle.load(f)

with open('book_titles.pkl', 'rb') as f:
    book_titles = pickle.load(f)

# Optional: metadata (images, authors, etc.)
try:
    with open('books_data.pkl', 'rb') as f:
        book_data = pickle.load(f)
except:
    book_data = None

# Title
st.title("Book Recommender System")
st.markdown("Get book recommendations based on what you like!")

# Dropdown for book selection
book_name = st.selectbox("Select a Book", book_titles)

# Number of recommendations
number = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

# Recommend button
if st.button("Recommend"):
    st.subheader("📖 Recommended Books:")
    
    # Find recommendations
    book_vector = matrix.loc[book_name].values.reshape(1, -1)
    distances, indices = model.kneighbors(book_vector, n_neighbors=number+1)

    # Display recommendations (excluding the selected book itself)
    for i in range(1, len(distances.flatten())):
        recommended_title = matrix.index[indices.flatten()[i]]
        st.markdown(f"**{i}. {recommended_title}**")

        # Optional: show additional info if available
        if book_data is not None:
            details = book_data[book_data['Book-Title'] == recommended_title].drop_duplicates('Book-Title')
            for _, row in details.iterrows():
                st.write(f"Author: {row['Book-Author']}")
                st.image(row['Image-URL-M'], width=120)
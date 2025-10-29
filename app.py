import streamlit as st
import requests

API_URL = st.secrets["API_URL"]


st.title("📚 Book & Research Paper Recommendation")

query = st.text_input("Enter Topic / Research Keyword")

top_n_books = st.number_input("Number of books", min_value=1, max_value=10, value=3)
top_n_papers = st.number_input("Number of research papers", min_value=1, max_value=10, value=2)

if st.button("Get Recommendations"):
    if not query:
        st.warning("Please enter a query")
    else:
        with st.spinner("Getting recommendations..."):
            response = requests.post(API_URL, data={
                "query": query,
                "top_n_books": top_n_books,
                "top_n_papers": top_n_papers,
            })

            if response.status_code == 200:
                result = response.json()

                st.subheader("📖 Recommended Books")
                for book in result.get("top_books", []):
                    st.write(f"✅ {book}")

                st.subheader("📄 Recommended Papers")
                for paper in result.get("top_papers", []):
                    st.write(f"🔍 {paper}")
            else:
                st.error("API error. Check backend logs.")

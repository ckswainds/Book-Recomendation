import streamlit as st
from src.models.model2.predict import start_prediction

from src.models.model1.predict import RecommenderPredictor
from src.entity.artifact_entity import BuildFeaturesArifact
from src.entity.config_entity import ModelTrainerConfig

st.set_page_config(page_title="AI Book & Paper Recommender", layout="wide")

# -------------------- Load CSS --------------------
with open("static/css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -------------------- TITLE --------------------
st.markdown("<h1>AI Book & Research Paper Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Find top-rated AI/ML books and research papers tailored to your query</p>", unsafe_allow_html=True)


# -------------------- INPUT FORM --------------------
with st.form(key="search_form"):
    st.markdown("<form>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    query = col1.text_input("Enter Topic (e.g., NLP, ML, Deep Learning)")
    top_n_books = col2.number_input("Top N Books", min_value=1, value=5)
    top_n_papers = col3.number_input("Top N Papers", min_value=1, value=5)

    submit = st.form_submit_button("🔍 Recommend")



# -------------------- PROCESS --------------------
if submit:
    st.session_state.result = start_prediction(query, top_n_books, top_n_papers)
    #Using tfidf
    # build_feat_artifact = BuildFeaturesArifact(
    #     modified_books_data_filepath="data/interim/modified_books.csv",
    #     modified_papers_data_filepath="data/interim/modified_papers.csv",
    # )
    # trainer_cfg = ModelTrainerConfig()
    # predictor = RecommenderPredictor(query, build_feat_artifact, trainer_cfg)
    # st.session_state.result = predictor.predict(top_books=top_n_books, top_papers=top_n_papers)


# -------------------- RESULTS DISPLAY --------------------
if "result" in st.session_state:
    result = st.session_state.result
    
    st.markdown(
        f"<h2>Recommendations for: <span style='color:#4338ca;'>{result['query']}</span></h2>",
        unsafe_allow_html=True
    )

    books = result.get("top_books", [])
    papers = result.get("top_papers", [])

    # ✅ BOOKS
    if books:
        st.markdown("<h3>📚 Top Books</h3>", unsafe_allow_html=True)
        cols = st.columns(3)

        for idx, book in enumerate(books):
            with cols[idx % 3]:
                st.markdown(
                    f"""
                    <div class="card fade-in">
                        <h3><a href="{book.get('previewLink','')}" target="_blank">{book.get('title','Unknown')}</a></h3>
                        <p><b>Author(s):</b> {book.get('authors','Unknown')}</p>
                        <p><b>Publisher:</b> {book.get('publisher','Unknown')}</p>
                        <p><b>Published:</b> {book.get('publishedDate','Unknown')}</p>
                        <p><b>Rating:</b> {book.get('avgrating','N/A')}</p>
                        <p class="line-clamp-4">{book.get('description','No description available')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ✅ PAPERS
    if papers:
        st.markdown("<h3>🧠 Top Research Papers</h3>", unsafe_allow_html=True)
        cols = st.columns(3)

        for idx, paper in enumerate(papers):
            with cols[idx % 3]:
                st.markdown(
                    f"""
                    <div class="card fade-in">
                        <h3><a href="{paper.get('URL','')}" target="_blank">{paper.get('Title','Unknown')}</a></h3>
                        <p><b>Authors:</b> {paper.get('Authors','Unknown')}</p>
                        <p><b>Year:</b> {paper.get('Year','Unknown')}</p>
                        <p><b>Citations:</b> {paper.get('Citations','0')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# -------------------- FOOTER --------------------
st.markdown("<footer>Developed with ❤️ | © 2025 AI Recommender</footer>", unsafe_allow_html=True)

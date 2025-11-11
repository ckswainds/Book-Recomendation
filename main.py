import os,sys
import streamlit as st
import json
import ast
# from app_src.entity.artifact_entity import BuildFeaturesArifact
# from app_src.entity.config_entity import ModelTrainerConfig
# from app_src.models.model1.predict import RecommenderPredictor
from app_src.logger import get_logger
from app_src.models.model2.predict import start_prediction
import os


# Configure page
st.set_page_config(
    page_title="AI Book & Research Paper Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)




# Custom CSS matching the screenshot with white-green gradient theme
st.markdown("""
    <style>
    /* Import Google Fonts for better typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Poppins:wght@400;500;600;700;800&display=swap');
    
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #e8f5e9 50%, #c8e6c9 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 1rem;
    }
    
    .main-header h1 {
        color: #6366f1;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-family: 'Poppins', sans-serif;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #475569;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Input container styling */
    .input-container {
        background: transparent;
        padding: 1rem 0;
        margin: 0 auto 2rem auto;
        max-width: 100%;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #d1d5db;
        padding: 0.85rem;
        font-size: 1rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4ade80;
        box-shadow: 0 0 0 3px rgba(74,222,128,0.1);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af;
        font-weight: 400;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #d1d5db;
        padding: 0.85rem;
        text-align: center;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border-radius: 8px;
        padding: 0.85rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        border: none;
        width: 100%;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(99,102,241,0.3);
        font-family: 'Poppins', sans-serif;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(99,102,241,0.4);
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.5rem;
        font-weight: 800;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding: 0.5rem 0;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Card styling */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        transition: all 0.3s;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.12);
        border-color: #4ade80;
    }
    
    .result-card h3 {
        color: #6366f1;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0 0 0.75rem 0;
        line-height: 1.4;
        font-family: 'Poppins', sans-serif;
    }
    
    .result-card h3 a {
        color: #6366f1;
        text-decoration: none;
        transition: color 0.3s;
        font-weight: 700;
    }
    
    .result-card h3 a:hover {
        color: #4f46e5;
        text-decoration: underline;
    }
    
    .card-meta {
        font-size: 0.9rem;
        color: #64748b;
        margin: 0.3rem 0;
        font-weight: 500;
        line-height: 1.6;
    }
    
    .card-meta strong {
        color: #1e293b;
        font-weight: 700;
    }
    
    .card-description {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-top: 0.75rem;
        flex-grow: 1;
        font-weight: 450;
    }
    
    .card-rating {
        display: inline-block;
        background: linear-gradient(90deg, #4ade80 0%, #22c55e 100%);
        color: white;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    /* Query display */
    .query-display {
        text-align: center;
        margin: 2rem 0 1.5rem 0;
    }
    
    .query-display h2 {
        color: #6366f1;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        font-family: 'Poppins', sans-serif;
    }
    
    .query-text {
        color: #1f2937;
        font-weight: 800;
        font-size: 1.3rem;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: #10b981 !important;
        color: white !important;
        padding: 1.25rem 2rem !important;
        font-weight: 800 !important;
        font-size: 1.25rem !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4) !important;
        text-align: center !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stSuccess > div {
        color: white !important;
        font-weight: 800 !important;
    }
    
    .stError {
        background-color: #ef4444 !important;
        color: white !important;
        padding: 1.25rem 2rem !important;
        font-weight: 800 !important;
        font-size: 1.25rem !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4) !important;
        text-align: center !important;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stError > div {
        color: white !important;
        font-weight: 800 !important;
    }
    
    /* Label styling */
    label {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #374151 !important;
    }
    
    /* Books and Papers labels */
    p {
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize logger
logger = get_logger(log_filename="app.log")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Header
st.markdown("""
    <div class="main-header">
        <h1>AI Book & Research Paper Recommender</h1>
        <p>Find top-rated books and research papers tailored to your query</p>
    </div>
""", unsafe_allow_html=True)

st.info(
    "📘 **Note:** Recommendations are currently available for selected topics in "
    "**Machine Learning, Deep Learning, NLP, Data Science, AI**, and core **Electronics** areas "
    "such as **Digital Electronics, Signal Processing, Communication Systems, VLSI, Control Systems.**"
)
# Input section
st.markdown('<div class="input-container">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([4, 1, 1])

with col1:
    query = st.text_input(
        "Search Query",
        placeholder="Enter your topic (e.g., machine learning, quantum physics, etc.)",
        label_visibility="collapsed",
        key="query_input"
    )

with col2:
    top_n_books = st.number_input(
        "Number of Books",
        min_value=1,
        max_value=10,
        value=5,
        label_visibility="collapsed",
        key="books_input"
    )
    st.markdown('<p style="text-align: center; font-size: 0.8rem; color: #6b7280; margin-top: -10px;">Books</p>', unsafe_allow_html=True)

with col3:
    top_n_papers = st.number_input(
        "Number of Papers",
        min_value=1,
        max_value=10,
        value=5,
        label_visibility="collapsed",
        key="papers_input"
    )
    st.markdown('<p style="text-align: center; font-size: 0.8rem; color: #6b7280; margin-top: -10px;">Papers</p>', unsafe_allow_html=True)

if st.button("Recommend", use_container_width=True):
    if not query.strip():
        st.error("⚠️ Please enter a search query")
    else:
        with st.spinner("🔄 Finding the best recommendations..."):
            try:
                # build_feat_artifact = BuildFeaturesArifact(
                #     modified_books_data_filepath="data/interim/modified_books.csv",
                #     modified_papers_data_filepath="data/interim/modified_papers.csv",
                # )
                # trainer_cfg = ModelTrainerConfig()
                
                # Using sentence transformer
                output_json = start_prediction(query, n_books=top_n_books, n_papers=top_n_papers)
                
                # Parse output
                if isinstance(output_json, dict):
                    output_obj = output_json
                elif isinstance(output_json, str):
                    try:
                        output_obj = json.loads(output_json)
                    except json.JSONDecodeError:
                        try:
                            output_obj = ast.literal_eval(output_json)
                        except Exception as e:
                            logger.exception("Failed to parse prediction string: %s", e)
                            raise
                else:
                    logger.warning("predictor.predict returned unexpected type: %s", type(output_json))
                    raise TypeError("Unexpected predictor return type")
                
                st.session_state.results = {
                    "query": query,
                    "top_books": output_obj.get("top_books", []),
                    "top_papers": output_obj.get("top_papers", []),
                }
                
                st.success("✅ Recommendations generated successfully!")
                
            except Exception as e:
                logger.exception("Prediction failed for query=%s: %s", query, e)
                st.error(f"❌ Prediction failed: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Display results
if st.session_state.results:
    results = st.session_state.results
    
    # Query display
    st.markdown(f"""
        <div class="query-display">
            <h2>Recommendations for: <span class="query-text">{results["query"]}</span></h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Books section
    if results["top_books"]:
        st.markdown('<div class="section-header">📚 Top Books</div>', unsafe_allow_html=True)
        
        # Create columns for grid layout
        cols_per_row = 3
        books = results["top_books"]
        
        for i in range(0, len(books), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(books):
                    book = books[i + j]
                    with col:
                        # Extract book information
                        title = book.get('title', book.get('Title', 'Unknown Title'))
                        author = book.get('authors', book.get('author', book.get('Author', 'Unknown')))
                        publisher = book.get('publisher', book.get('Publisher', 'N/A'))
                        published_date = book.get('publishedDate', book.get('published_date', book.get('Published', 'N/A')))
                        rating = book.get('avgrating', book.get('rating', book.get('Rating', book.get('score', 'N/A'))))
                        description = book.get('description', book.get('Description', book.get('summary', '')))
                        link = book.get('previewLink', book.get('link', book.get('url', book.get('infoLink', ''))))
                        
                        # Create title HTML
                        if link and link.strip():
                            title_html = f'<h3><a href="{link}" target="_blank">{title}</a></h3>'
                        else:
                            title_html = f'<h3>{title}</h3>'
                        
                        # Truncate description
                        desc_html = ''
                        if description:
                            desc_text = description[:200] + '...' if len(description) > 200 else description
                            desc_html = f'<div class="card-description">{desc_text}</div>'
                        
                        st.markdown(f"""
                            <div class="result-card">
                                {title_html}
                                <div class="card-meta"><strong>Author(s):</strong> {author}</div>
                                <div class="card-meta"><strong>Publisher:</strong> {publisher}</div>
                                <div class="card-meta"><strong>Published:</strong> {published_date}</div>
                                <div class="card-meta"><strong>Rating:</strong> <span class="card-rating">{rating}</span></div>
                                {desc_html}
                            </div>
                        """, unsafe_allow_html=True)
    
    # Papers section
    if results["top_papers"]:
        st.markdown('<div class="section-header">🔬 Top Research Papers</div>', unsafe_allow_html=True)
        
        papers = results["top_papers"]
        
        for i in range(0, len(papers), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(papers):
                    paper = papers[i + j]
                    with col:
                        # Extract paper information
                        title = paper.get('Title', paper.get('title', 'Unknown Title'))
                        authors = paper.get('Authors', paper.get('authors', paper.get('author', 'Unknown')))
                        year = paper.get('Year', paper.get('year', 'N/A'))
                        citations = paper.get('Citations', paper.get('citations', paper.get('citation_count', 'N/A')))
                        abstract = paper.get('abstract', paper.get('Abstract', paper.get('summary', '')))
                        link = paper.get('URL', paper.get('url', paper.get('link', paper.get('doi', paper.get('arxiv_url', '')))))
                        
                        # Create title HTML
                        if link and link.strip():
                            title_html = f'<h3><a href="{link}" target="_blank">{title}</a></h3>'
                        else:
                            title_html = f'<h3>{title}</h3>'
                        
                        # Truncate abstract
                        abstract_html = ''
                        if abstract:
                            abstract_text = abstract[:200] + '...' if len(abstract) > 200 else abstract
                            abstract_html = f'<div class="card-description">{abstract_text}</div>'
                        
                        st.markdown(f"""
                            <div class="result-card">
                                {title_html}
                                <div class="card-meta"><strong>Authors:</strong> {authors}</div>
                                <div class="card-meta"><strong>Year:</strong> {year}</div>
                                <div class="card-meta"><strong>Citations:</strong> <span class="card-rating">{citations}</span></div>
                                {abstract_html}
                            </div>
                        """, unsafe_allow_html=True)
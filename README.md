Book_recomendation
==============================

# AI Book & Research Paper Recommender

![Project Banner](https://img.shields.io/badge/ML-Content--Based-blue) ![Python](https://img.shields.io/badge/Python-3.11-green) ![FastAPI](https://img.shields.io/badge/FastAPI-API-red)

A **Machine Learning-powered content-based recommendation system** that suggests top books and research papers based on user queries. The system fetches data from **Google Books API** and **Semantic Scholar API**, processes it, and provides ranked recommendations.

---

## 🔍 Features

- Recommend **top N books** and **top N research papers** for any AI, ML, NLP, or Electronics-related topic.
- Uses **Sentence Transformer for embeddings and  content similarity** for content-based ranking.
- Considers multiple metadata features:
  - **Books:** Title, Authors, Publisher, Published Date, Average Rating, Preview Link
  - **Papers:** Title, Authors, Year, Citations, URL
- Built with **FastAPI** + **TailwindCSS** front-end for a responsive web interface.
- Fully reproducible pipeline for **data ingestion, preprocessing, feature building, and model training**.

---
Project Organization
------------

```
BOOK-RECOMENDATION/
├── LICENSE     
├── README.md                  
├── Makefile                     # Makefile with commands like `make data` or `make train`                   
├── configs                      # Config files (models and training hyperparameters)
│   └── model1.yaml              
│
├── data                         
│   ├── external                 # Data from third party sources.
│   ├── interim                  # Intermediate data that has been transformed.
│   ├── processed                # The final, canonical data sets for modeling.
│   └── raw                      # The original, immutable data dump.
│
├── docs                         # Project documentation.
│
├── models                       # Trained and serialized models.
│
├── notebooks                    # Jupyter notebooks.
│
├── references                   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                  # Generated graphics and figures to be used in reporting.
│
├── requirements.txt             # The requirements file for reproducing the analysis environment.
└── src                          # Source code for use in this project.
    ├── __init__.py              # Makes src a Python module.
    │
    ├── data                     # Data engineering scripts.
    │   ├── build_features.py    
    │   ├── cleaning.py          
    │   ├── ingestion.py         
    │   ├── labeling.py          
    │   ├── splitting.py         
    │   └── validation.py        
    │
    ├── models                   # ML model engineering (a folder for each model).
    │   └── model1      
    │       ├── dataloader.py    
    │       ├── hyperparameters_tuning.py 
    │       ├── model.py         
    │       ├── predict.py       
    │       ├── preprocessing.py 
    │       └── train.py         
    │
    └── visualization        # Scripts to create exploratory and results oriented visualizations.
        ├── evaluation.py        
        └── exploration.py       
```


--------
<p><small>Project based on the <a target="_blank" href="https://github.com/Chim-SO/cookiecutter-mlops/">cookiecutter MLOps project template</a>
that is originally based on <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 
#cookiecuttermlops #cookiecutterdatascience</small></p>

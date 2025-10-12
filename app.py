from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.entity.artifact_entity import BuildFeaturesArifact
from src.entity.config_entity import ModelTrainerConfig
from src.models.model1.predict import RecommenderPredictor
from src.logger import get_logger
import json
import ast

# The application instance must be named 'app' for the Docker CMD to find it: app:app
app = FastAPI() 
logger = get_logger(log_filename="app.log")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Handles the root path, serving the main index.html template."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    query: str = Form(...),
    top_n_books: int = Form(3),
    top_n_papers: int = Form(2),
):
    """
    Handles the prediction request from the user form. 
    It takes a query and returns book and paper recommendations.
    """
    build_feat_artifact = BuildFeaturesArifact(
        modified_books_data_filepath="data/interim/modified_books.csv",
        modified_papers_data_filepath="data/interim/modified_papers.csv",
    )
    trainer_cfg = ModelTrainerConfig()

    try:
        predictor = RecommenderPredictor(query, build_feat_artifact, trainer_cfg)
        output_json = predictor.predict(top_books=top_n_books, top_papers=top_n_papers)

        # Accept dict or JSON string (some predictor implementations returned str)
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
            # unexpected type; coerce if possible or fail gracefully
            logger.warning("predictor.predict returned unexpected type: %s", type(output_json))
            raise TypeError("Unexpected predictor return type")

        result = {
            "query": query,
            "top_books": output_obj.get("top_books", []),
            "top_papers": output_obj.get("top_papers", []),
        }

    except Exception as e:
        logger.exception("Prediction failed for query=%s: %s", query, e)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Prediction failed: {str(e)}",
            },
        )

    return templates.TemplateResponse("index.html", {"request": request, "result": result})
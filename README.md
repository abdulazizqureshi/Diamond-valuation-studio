# Diamond Valuation Studio — End-to-End MLOps

An end-to-end machine learning web application that predicts diamond prices based on physical dimensions and gemological attributes (carat, cut, color, clarity, depth, and table). Built with a modular Python package architecture, Scikit-Learn pipelines, DVC, MLflow tracking, Docker, and an interactive 3D Three.js + Flask web interface.

---

## Project Architecture

- **`src/DiamondPricePrediction/components/`**: Modular stages for data ingestion, feature transformation, model training, and evaluation.
- **`src/DiamondPricePrediction/pipelines/`**: End-to-end training and inference pipelines.
- **`Artifacts/`**: Serialized preprocessor (`preprocessor.pkl`), trained Lasso regression model (`model.pkl`, $R^2 \approx 0.936$), and processed datasets.
- **`app.py`, `templates/`, `static/`**: Flask web server and interactive 3D Three.js frontend for real-time price predictions.
- **`dvc.yaml` & `Dockerfile`**: Reproducible pipeline orchestration and containerization support.

---

## Tech Stack

- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **MLOps & Tracking:** MLflow, DVC, Docker
- **Frontend & Backend:** Flask, Three.js, HTML5, CSS3

---

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/abdulazizqureshi/diamond-valuation-studio.git](https://github.com/abdulazizqureshi/diamond-valuation-studio.git)
   cd diamond-valuation-studio
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Start the Flask web application:**
   ```bash
   flask --app app run --host=127.0.0.1 --port=5050
   ```

5. **Open in browser:** Navigate to `http://127.0.0.1:5050` and submit diamond measurements to generate a price estimate.
Data source :
https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv
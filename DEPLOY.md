# Deployment Instructions

## Streamlit Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New App**
4. Repository: `B2prakash/fifa-win-predictor`
5. Branch: `main`
6. Main file path: `app/streamlit_app.py`
7. Click **Deploy**

## Local Run

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Requirements

- Python 3.9+
- All packages listed in `requirements.txt`

## Project Structure

```
fifa-win-predictor/
├── app/
│   └── streamlit_app.py      # Main Streamlit app (4 pages)
├── src/
│   ├── preprocess.py         # Data cleaning pipeline
│   ├── features.py           # Feature engineering
│   └── model.py              # Model training & evaluation
├── data/
│   ├── WorldCupMatches.csv   # Historical WC match data
│   ├── fifa_ranking.csv      # FIFA world rankings
│   ├── features.csv          # Engineered features
│   ├── fixtures_2026.csv     # 2026 fixture predictions
│   ├── results.csv           # International results (martj42)
│   └── star_players_2026.csv # Star player availability
├── models/
│   ├── fifa_model.pkl        # Trained model (Logistic Regression)
│   └── shap_importance.png   # SHAP feature importance plot
├── .streamlit/
│   └── config.toml           # Streamlit theme & server config
├── requirements.txt
└── README.md
```

## Model Details

- **Algorithm**: Logistic Regression (best AUC among LR, RF, XGBoost)
- **Features**: 15 (rank diff, win rates, H2H, form, confederation strength, coach experience)
- **AUC-ROC**: 0.888
- **Training data**: World Cup 1994–2022 + FIFA Rankings

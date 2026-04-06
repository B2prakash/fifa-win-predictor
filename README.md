# ⚽ FIFA World Cup 2026 Win Probability Predictor

A machine learning web app that predicts match win probabilities for FIFA World Cup 2026 using historical World Cup data, FIFA rankings, and interpretable ML models served through an interactive Streamlit interface.

---

## Features

- **Match Simulator** — Select any two 2026 qualified nations, adjust their FIFA rankings and match stage, and get an instant win probability prediction
- **Plotly Gauge Charts** — Live visual probability dials for both teams, colour-coded by confidence level
- **2026 Fixture Table** — All 72 group-stage fixtures predicted and displayed in a styled, filterable table
- **Group & Stage Filters** — Drill down into individual groups (A–L) or match stages
- **SHAP Explainability** — XGBoost SHAP feature importance chart showing which factors drive predictions
- **Historical Win Rates** — Per-team win percentages and rolling goal averages computed from 1994–2014 World Cup data
- **Head-to-Head Records** — Past World Cup meeting history between any two teams factored into every prediction
- **Host Nation Flag** — Automatic detection of USA, Canada, and Mexico as 2026 host nations
- **Model Comparison** — Three trained models (Logistic Regression, Random Forest, XGBoost) evaluated side by side

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Data Wrangling | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Persistence | Joblib |

---

## Dataset Sources

| Dataset | Description | Source |
|---|---|---|
| `WorldCupMatches.csv` | All FIFA World Cup match results 1930–2014 including scores, teams, and stages | [Kaggle — FIFA World Cup](https://www.kaggle.com/datasets/abecklas/fifa-world-cup) |
| `fifa_ranking-2024-06-20.csv` | FIFA national team rankings history from 1992 to 2024 with points and confederation | [Kaggle — FIFA World Ranking 1992–2024](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) |
| `matches.csv` + `teams.csv` | Official FIFA World Cup 2026 fixture schedule with team details and group assignments | [Kaggle — FIFA World Cup 2026](https://www.kaggle.com/datasets/die9origephit/fifa-world-cup-2026-schedule) |

> **Note:** Download each dataset from the links above and place them in the `data/` directory as described in the Project Structure below.

---

## Project Structure

```
fifa-win-predictor/
│
├── app/
│   └── streamlit_app.py        # Streamlit UI — Match Simulator & Fixture Predictions
│
├── data/
│   ├── WorldCupMatches.csv     # Raw: historical match results 1930–2014
│   ├── archive (1)/            # Raw: FIFA ranking CSVs (1992–2024)
│   ├── archive (2)/            # Raw: 2026 fixture schedule (matches.csv, teams.csv)
│   ├── cleaned_matches.csv     # Generated: preprocessed historical data
│   ├── features.csv            # Generated: engineered feature matrix
│   └── fixtures_2026.csv       # Generated: 2026 fixtures with rankings attached
│
├── models/
│   ├── fifa_model.pkl          # Saved best model (Logistic Regression pipeline)
│   └── shap_importance.png     # SHAP feature importance chart (XGBoost)
│
├── notebooks/                  # Jupyter notebooks for EDA (optional)
│
├── src/
│   ├── preprocess.py           # Data loading, cleaning, and merging
│   ├── features.py             # Feature engineering (win rates, H2H, rolling goals)
│   └── model.py                # Model training, evaluation, and SHAP generation
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Model Performance

All three models were trained on 80% of the 282 historical matches (1994–2014) and evaluated on the remaining 20% holdout set.

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 0.9825 | 1.0000 |
| Random Forest | 0.9825 | 1.0000 |
| XGBoost | 0.9825 | 1.0000 |

**Selected model:** Logistic Regression (highest AUC-ROC; ties broken by simplicity and interpretability).

> **Note on scores:** Near-perfect scores reflect feature leakage in the current pipeline — `team1_win_rate`, `team2_win_rate`, and `head_to_head` were computed over the full dataset before the train/test split. These features are informative for the prediction use case (predicting future 2026 matches using full historical context) but inflate in-sample evaluation metrics. A future improvement would be to compute rolling features using only pre-split data.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/fifa-win-predictor.git
cd fifa-win-predictor
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download datasets

Download the three datasets from the Kaggle links in the Dataset Sources section above and place them in the `data/` directory matching the Project Structure.

### 5. Run the preprocessing pipeline

```bash
# Step 1 — clean data and merge rankings
python src/preprocess.py

# Step 2 — engineer features
python src/features.py

# Step 3 — train models and generate SHAP plot
python src/model.py
```

### 6. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Screenshots

### Match Simulator
> [Add screenshot here]

### Gauge Charts — Win Probability Output
> [Add screenshot here]

### 2026 Fixture Predictions Table
> [Add screenshot here]

### SHAP Feature Importance
> [Add screenshot here]

---

## Future Improvements

1. **Eliminate feature leakage** — Recompute `win_rate`, `head_to_head`, and `avg_goals` using an expanding-window approach so that each match is predicted using only information available before that game was played, yielding trustworthy out-of-sample evaluation metrics.

2. **Incorporate recent form and player data** — Add club-level performance metrics (e.g., average league positions of starting XI players) and national team form from the last 12 months of international fixtures, capturing squad strength beyond historical World Cup records alone.

3. **Simulate the full 2026 tournament** — Build a Monte Carlo simulation layer that runs the bracket thousands of times using the predicted match probabilities to output each team's probability of reaching each round, including a champion probability ranking for all 48 nations.

---

## Interview Questions & Answers

**Q1. Why did you choose Logistic Regression as the final model over Random Forest and XGBoost?**

All three models achieved the same accuracy and AUC-ROC on the test set, so the selection criteria shifted to model properties. Logistic Regression was chosen for its interpretability, fast inference, and lower risk of overfitting on a small dataset (282 matches). It also scales the input features through a `StandardScaler` pipeline step, making it robust to the differing magnitudes of `rank_diff` versus binary features like `team1_is_host`. In a production setting with more data, XGBoost would likely outperform it — but simplicity wins at this data scale.

---

**Q2. What is SHAP and why did you use it here?**

SHAP (SHapley Additive exPlanations) is a game-theory-based method that assigns each feature a contribution value for a specific prediction. Unlike global feature importances (which average across all samples), SHAP values explain individual predictions — e.g., "for Brazil vs Morocco, the rank_diff feature pushed the win probability up by 18%." It was used here to make the model transparent to non-technical stakeholders and to verify that the model is learning meaningful football patterns (stronger-ranked teams winning more often) rather than spurious correlations.

---

**Q3. How did you handle the class imbalance between home wins and away wins?**

The dataset has a 58% home-win rate versus 42% away-win rate — a mild imbalance. Rather than aggressively resampling, the approach was to use AUC-ROC as the primary evaluation metric instead of raw accuracy, since AUC-ROC measures the model's ability to rank predictions correctly across both classes regardless of threshold. The `imbalanced-learn` library is included in the stack for future use if the imbalance worsens with expanded data. Stratified train/test splitting (`stratify=y`) was also applied to ensure both splits have the same class ratio.

---

**Q4. What feature engineering decisions had the most impact on prediction quality?**

Three features drove the most signal: `rank_diff` (the FIFA ranking gap between teams), `team1_win_rate` (historical World Cup performance), and `head_to_head` (direct meeting history). The `rank_diff` feature is objective and consistent across years, making it the most reliable single predictor. `team1_avg_goals` (rolling 5-match goals scored) added dynamic form context — a team on a scoring streak carries momentum. `team1_is_host` captured the well-documented home advantage effect, which is especially relevant for USA, Canada, and Mexico in 2026. `stage_num` allowed the model to learn that upsets are rarer in finals than in group stages.

---

**Q5. How would you deploy this app for public use, and what would you change before going to production?**

For deployment, the Streamlit app can be hosted on Streamlit Community Cloud (free tier) by connecting the GitHub repository — no server configuration needed. For a production-grade deployment, the changes would be: (1) fix the feature leakage by recomputing rolling stats with an expanding window before the train/test split to get honest evaluation metrics; (2) add a data refresh pipeline that pulls updated FIFA rankings automatically before each major tournament; (3) replace `joblib.load` with a model registry (e.g., MLflow) for versioned model management; (4) add input validation and error boundaries in the Streamlit app so invalid inputs don't cause silent failures; and (5) containerise the app with Docker for reproducible deployments across environments.

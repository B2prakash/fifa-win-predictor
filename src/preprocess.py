import pandas as pd
import numpy as np
import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
FIX_DIR    = os.path.join(DATA_DIR, "archive (2)")

WC_PATH    = os.path.join(DATA_DIR, "WorldCupMatches.csv")
RANK_PATH  = os.path.join(DATA_DIR, "fifa_ranking.csv")
MATCH_PATH = os.path.join(DATA_DIR, "wc2026_fixtures.csv")
TEAMS_PATH = os.path.join(FIX_DIR,  "teams.csv")

OUT_HIST   = os.path.join(DATA_DIR, "cleaned_matches.csv")
OUT_FIX    = os.path.join(DATA_DIR, "fixtures_2026.csv")

# ── Name normalisation map (WC name → FIFA ranking name) ─────────────────────
NAME_MAP = {
    "Germany FR":               "Germany",
    "German DR":                "Germany",
    "Czech Republic":           "Czechia",
    "Soviet Union":             "Russia",
    "Dutch East Indies":        "Indonesia",
    "Iran":                     "IR Iran",
    "China PR":                 "China PR",
    # HTML-garbled names in the CSV
    'rn">Bosnia and Herzegovina':  "Bosnia and Herzegovina",
    'rn">Republic of Ireland':     "Republic of Ireland",
    'rn">Serbia and Montenegro':   "Serbia and Montenegro",
    'rn">Trinidad and Tobago':     "Trinidad and Tobago",
    'rn">United Arab Emirates':    "United Arab Emirates",
    "C\ufffdte d'Ivoire":          "Côte d'Ivoire",
    "Côte d'Ivoire":               "Côte d'Ivoire",
}

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading WorldCupMatches.csv …")
wc = pd.read_csv(WC_PATH)

print("Loading FIFA rankings …")
rank = pd.read_csv(RANK_PATH)

print("Loading 2026 fixtures …")
matches26 = pd.read_csv(MATCH_PATH)
teams26   = pd.read_csv(TEAMS_PATH)

# ── 2. Clean WorldCupMatches ──────────────────────────────────────────────────
# Strip whitespace from string columns
for col in wc.select_dtypes("object").columns:
    wc[col] = wc[col].str.strip()

# Fix encoding / legacy names
wc["HomeTeamName"] = wc["HomeTeamName"].replace(NAME_MAP)
wc["AwayTeamName"] = wc["AwayTeamName"].replace(NAME_MAP)

# Keep only columns we need
wc = wc[["Year", "Stage", "HomeTeamName", "AwayTeamName",
          "HomeTeamGoals", "AwayTeamGoals", "WinConditions"]].copy()

# Drop rows missing goals (can't determine result)
wc.dropna(subset=["HomeTeamGoals", "AwayTeamGoals"], inplace=True)
wc["HomeTeamGoals"] = wc["HomeTeamGoals"].astype(int)
wc["AwayTeamGoals"] = wc["AwayTeamGoals"].astype(int)
wc["Year"] = wc["Year"].astype(int)

# ── 3. Clean FIFA ranking ─────────────────────────────────────────────────────
rank["rank_date"] = pd.to_datetime(rank["rank_date"])
rank["year"]      = rank["rank_date"].dt.year
rank.dropna(subset=["rank", "country_full"], inplace=True)
rank["rank"] = rank["rank"].astype(int)

# For each team keep only the last ranking snapshot of each year
rank_annual = (
    rank.sort_values("rank_date")
        .groupby(["country_full", "year"], as_index=False)
        .last()[["country_full", "year", "rank", "total_points", "confederation"]]
)

# ── 4. Merge historical matches with FIFA rankings ────────────────────────────
# We join on (team_name, ranking_year) where ranking_year is the latest year
# ≤ the World Cup year that has data (FIFA rankings only go back to 1992;
# matches before that will have NaN rank values).

def merge_ranking(df, team_col, rank_col_prefix):
    """Left-join df with rank_annual on team name and closest prior year."""
    merged = df.merge(
        rank_annual.rename(columns={
            "country_full": team_col,
            "year":         f"{rank_col_prefix}_rank_year",
            "rank":         f"{rank_col_prefix}_rank",
            "total_points": f"{rank_col_prefix}_points",
            "confederation":f"{rank_col_prefix}_conf",
        }),
        on=team_col,
        how="left",
    )
    # Keep only rows where ranking year ≤ WC year, then take the closest one
    merged = merged[merged[f"{rank_col_prefix}_rank_year"] <= merged["Year"]]
    merged = (
        merged.sort_values(f"{rank_col_prefix}_rank_year")
              .groupby(list(df.columns), as_index=False)
              .last()
    )
    merged.drop(columns=[f"{rank_col_prefix}_rank_year"], inplace=True)
    return merged

wc = merge_ranking(wc, "HomeTeamName", "home")
wc = merge_ranking(wc, "AwayTeamName", "away")

# ── 5. Remove draws ───────────────────────────────────────────────────────────
wc = wc[wc["HomeTeamGoals"] != wc["AwayTeamGoals"]].copy()

# ── 6. Create binary target ───────────────────────────────────────────────────
wc["home_win"] = (wc["HomeTeamGoals"] > wc["AwayTeamGoals"]).astype(int)

# Rank difference: positive → home team is ranked higher (lower rank number = better)
wc["rank_diff"] = wc["away_rank"] - wc["home_rank"]   # positive = home advantage

# ── 7. Final tidy-up ──────────────────────────────────────────────────────────
wc.reset_index(drop=True, inplace=True)

# ── 8. Save cleaned historical data ──────────────────────────────────────────
wc.to_csv(OUT_HIST, index=False)
print(f"\nSaved: {OUT_HIST}")
print(f"Shape: {wc.shape}")
print(wc.head())

# ── 9. Build & clean 2026 fixtures ───────────────────────────────────────────
# Only group-stage matches with both teams known (no placeholders / TBDs)
fix = matches26.merge(
    teams26[["id", "team_name", "fifa_code", "group_letter", "is_placeholder"]]
              .rename(columns={"id": "home_team_id",
                               "team_name": "HomeTeamName",
                               "fifa_code": "home_fifa_code",
                               "group_letter": "group",
                               "is_placeholder": "home_is_placeholder"}),
    on="home_team_id", how="left"
).merge(
    teams26[["id", "team_name", "fifa_code", "is_placeholder"]]
              .rename(columns={"id": "away_team_id",
                               "team_name": "AwayTeamName",
                               "fifa_code": "away_fifa_code",
                               "is_placeholder": "away_is_placeholder"}),
    on="away_team_id", how="left"
)

# Drop knockout/TBD rows (missing team ids) and placeholder teams
fix = fix.dropna(subset=["home_team_id", "away_team_id"])
fix = fix[~fix["home_is_placeholder"] & ~fix["away_is_placeholder"]]

# Keep useful columns
fix = fix[["match_number", "HomeTeamName", "AwayTeamName",
           "home_fifa_code", "away_fifa_code", "group",
           "kickoff_at", "match_label"]].copy()

# Add latest FIFA rankings (2024 snapshot)
latest_rank = (
    rank_annual.sort_values("year")
               .groupby("country_full", as_index=False)
               .last()[["country_full", "rank", "total_points", "confederation"]]
)

# Normalise 2026 team names to match FIFA ranking names
FIX_NAME_MAP = {
    "Côte d'Ivoire": "Côte d'Ivoire",
    "Curaçao": "Curaçao",
}
fix["HomeTeamName"] = fix["HomeTeamName"].replace(FIX_NAME_MAP)
fix["AwayTeamName"] = fix["AwayTeamName"].replace(FIX_NAME_MAP)

fix = fix.merge(latest_rank.rename(columns={
    "country_full": "HomeTeamName",
    "rank": "home_rank", "total_points": "home_points",
    "confederation": "home_conf"}), on="HomeTeamName", how="left")

fix = fix.merge(latest_rank.rename(columns={
    "country_full": "AwayTeamName",
    "rank": "away_rank", "total_points": "away_points",
    "confederation": "away_conf"}), on="AwayTeamName", how="left")

fix["rank_diff"] = fix["away_rank"] - fix["home_rank"]
fix.reset_index(drop=True, inplace=True)

# ── 10. Save 2026 fixtures ────────────────────────────────────────────────────
fix.to_csv(OUT_FIX, index=False)
print(f"\nSaved: {OUT_FIX}")
print(f"Shape: {fix.shape}")
print(fix.head())

import pandas as pd
import numpy as np
import os

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
IN_PATH      = os.path.join(DATA_DIR, "cleaned_matches.csv")
OUT_PATH     = os.path.join(DATA_DIR, "features.csv")
RESULTS_PATH = os.path.join(DATA_DIR, "results.csv")

# ── Name map: cleaned_matches names → results.csv names ──────────────────────
RESULTS_NAME_MAP = {
    "Côte d'Ivoire":          "Ivory Coast",
    "IR Iran":                "Iran",
    "Korea DPR":              "North Korea",
    "Korea Republic":         "South Korea",
    "Serbia and Montenegro":  "Serbia",   # dissolved 2006; use Serbia proxy
    "USA":                    "United States",
}

# ── Stage → numeric difficulty level ─────────────────────────────────────────
STAGE_MAP = {
    **{f"Group {g}": 0 for g in "ABCDEFGH"},
    "Round of 32":            1,
    "Round of 16":            2,
    "Quarter-finals":         3,
    "Semi-finals":            4,
    "Third place":            4,
    "Match for third place":  4,
    "Play-off for third place": 4,
    "Final":                  5,
}

HOSTS = {
    1994: ["USA"],
    1998: ["France"],
    2002: ["South Korea", "Japan"],
    2006: ["Germany"],
    2010: ["South Africa"],
    2014: ["Brazil"],
    2018: ["Russia"],
    2022: ["Qatar"],
    2026: ["USA", "Canada", "Mexico"],
}

# ── Confederation map ─────────────────────────────────────────────────────────
CONFEDERATION_MAP = {
    # UEFA
    "Germany": "UEFA", "France": "UEFA", "Spain": "UEFA", "Italy": "UEFA",
    "England": "UEFA", "Portugal": "UEFA", "Netherlands": "UEFA",
    "Belgium": "UEFA", "Croatia": "UEFA", "Switzerland": "UEFA",
    "Denmark": "UEFA", "Poland": "UEFA", "Sweden": "UEFA", "Serbia": "UEFA",
    "Austria": "UEFA", "Scotland": "UEFA", "Wales": "UEFA",
    "Czech Republic": "UEFA", "Czechia": "UEFA", "Hungary": "UEFA",
    "Slovakia": "UEFA", "Slovenia": "UEFA", "Albania": "UEFA",
    "Turkey": "UEFA", "Romania": "UEFA", "Ukraine": "UEFA",
    "Greece": "UEFA", "Bosnia and Herzegovina": "UEFA", "Bulgaria": "UEFA",
    "Norway": "UEFA", "Republic of Ireland": "UEFA", "Russia": "UEFA",
    "Yugoslavia": "UEFA", "Serbia and Montenegro": "UEFA",
    "North Macedonia": "UEFA", "Finland": "UEFA", "Iceland": "UEFA",
    "Luxembourg": "UEFA", "Kosovo": "UEFA", "Montenegro": "UEFA",
    # CONMEBOL
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Peru": "CONMEBOL", "Ecuador": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # AFC
    "Japan": "AFC", "Korea Republic": "AFC", "South Korea": "AFC",
    "IR Iran": "AFC", "Saudi Arabia": "AFC", "Australia": "AFC",
    "China PR": "AFC", "Iraq": "AFC", "Qatar": "AFC", "Uzbekistan": "AFC",
    "Jordan": "AFC", "Indonesia": "AFC", "Korea DPR": "AFC",
    "United Arab Emirates": "AFC", "Oman": "AFC", "Bahrain": "AFC",
    "Kuwait": "AFC", "Thailand": "AFC", "Vietnam": "AFC",
    # CAF
    "Morocco": "CAF", "Senegal": "CAF", "Nigeria": "CAF", "Ghana": "CAF",
    "Cameroon": "CAF", "Egypt": "CAF", "Tunisia": "CAF", "Algeria": "CAF",
    "Côte d'Ivoire": "CAF", "Mali": "CAF", "South Africa": "CAF",
    "Angola": "CAF", "Togo": "CAF", "Cabo Verde": "CAF",
    "DR Congo": "CAF", "Tanzania": "CAF", "Comoros": "CAF", "Benin": "CAF",
    "Zambia": "CAF", "Burkina Faso": "CAF", "Uganda": "CAF",
    "Zimbabwe": "CAF", "Cape Verde": "CAF",
    # CONCACAF
    "USA": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF", "Panama": "CONCACAF",
    "Honduras": "CONCACAF", "El Salvador": "CONCACAF", "Haiti": "CONCACAF",
    "Curaçao": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "Guatemala": "CONCACAF", "Cuba": "CONCACAF",
    # OFC
    "New Zealand": "OFC", "Australia": "AFC",   # Australia moved to AFC in 2006
}

# ── Confederation historical WC win rates 1930–2022 ───────────────────────────
CONF_STRENGTH = {
    "CONMEBOL": 0.58,
    "UEFA":     0.54,
    "CONCACAF": 0.38,
    "AFC":      0.35,
    "CAF":      0.34,
    "OFC":      0.28,
}

# ── Coach World Cup experience (head-coach appearances at a WC, as of 2026) ───
# Source: manually compiled from public records.
# Applies as a team-level static feature; unknown teams default to 0.
COACH_WC_EXP = {
    # CONMEBOL
    "Argentina":              1,  # Lionel Scaloni: 2022 (won)
    "Brazil":                 0,  # Dorival Júnior: debut 2026
    "Uruguay":                1,  # Marcelo Bielsa coached Chile 2010
    "Colombia":               0,  # Néstor Lorenzo: debut 2026
    "Ecuador":                0,  # Sebastián Beccacece: debut 2026
    "Paraguay":               1,  # Gustavo Alfaro: Ecuador 2022
    "Chile":                  1,  # Ricardo Gareca: Peru 2018
    "Bolivia":                0,
    "Peru":                   0,
    "Venezuela":              0,
    # UEFA
    "France":                 3,  # Didier Deschamps: 2014, 2018 (won), 2022
    "Spain":                  0,  # Luis de la Fuente: debut 2026
    "Germany":                0,  # Julian Nagelsmann: debut 2026
    "England":                0,  # Thomas Tuchel: debut 2026
    "Portugal":               1,  # Roberto Martínez: Belgium 2022
    "Netherlands":            1,  # Ronald Koeman: 2014
    "Croatia":                2,  # Zlatko Dalić: 2018, 2022
    "Belgium":                0,  # Domenico Tedesco: debut 2026
    "Switzerland":            0,  # Murat Yakin: debut 2026
    "Serbia":                 1,  # Dragan Stojković: 2022
    "Denmark":                0,
    "Austria":                0,  # Ralf Rangnick: debut 2026
    "Turkey":                 0,  # Vincenzo Montella: debut 2026
    "Ukraine":                0,
    "Poland":                 0,
    "Hungary":                0,
    "Slovakia":               0,
    "Slovenia":               0,
    "Scotland":               0,
    "Romania":                0,
    "Greece":                 0,
    "Sweden":                 0,
    "Norway":                 0,
    "Bulgaria":               0,
    "Italy":                  0,
    "Russia":                 0,
    "Yugoslavia":             0,
    "Serbia and Montenegro":  0,
    "Bosnia and Herzegovina": 0,
    "Republic of Ireland":    0,
    # CONCACAF
    "USA":                    0,  # Mauricio Pochettino: debut 2026
    "Mexico":                 2,  # Javier Aguirre: 2002, 2010
    "Canada":                 0,  # Jesse Marsch: debut 2026
    "Costa Rica":             0,
    "Panama":                 0,
    "Honduras":               0,
    "Jamaica":                0,
    "Haiti":                  0,
    "Trinidad and Tobago":    0,
    # AFC
    "Japan":                  1,  # Hajime Moriyasu: 2022
    "Korea Republic":         1,  # Hong Myung-bo: 2014
    "South Korea":            1,  # same team, alternate name
    "Australia":              0,  # Tony Popovic: debut 2026
    "IR Iran":                0,  # Amir Ghalenoei: debut 2026
    "Saudi Arabia":           0,
    "Jordan":                 0,
    "Uzbekistan":             0,
    "Korea DPR":              0,
    "China PR":               0,
    "Indonesia":              0,
    "Qatar":                  0,
    # CAF
    "Morocco":                1,  # Walid Regragui: 2022
    "Senegal":                1,  # Aliou Cissé: 2022
    "Nigeria":                0,
    "South Africa":           0,
    "Egypt":                  0,
    "Algeria":                0,
    "Cameroon":               0,
    "Ghana":                  0,
    "Côte d'Ivoire":          0,
    "Tunisia":                0,
    "Togo":                   0,
    "Angola":                 0,
    "Cabo Verde":             0,
    "Curaçao":                0,
    "New Zealand":            0,
}

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(IN_PATH)
df = df.sort_values("Year").reset_index(drop=True)

# ── Build long-form match history ─────────────────────────────────────────────
home_view = df[["Year", "HomeTeamName", "HomeTeamGoals", "home_win"]].rename(
    columns={"HomeTeamName": "team", "HomeTeamGoals": "goals_scored", "home_win": "win"}
)
away_view = df[["Year", "AwayTeamName", "AwayTeamGoals", "home_win"]].rename(
    columns={"AwayTeamName": "team", "AwayTeamGoals": "goals_scored"}
)
away_view["win"] = 1 - df["home_win"].values

long = pd.concat([home_view, away_view], ignore_index=True).sort_values(
    ["team", "Year"]
).reset_index(drop=True)

# ── Feature 1: rank_diff ──────────────────────────────────────────────────────
df["rank_diff"] = df["home_rank"] - df["away_rank"]

# ── Features 2 & 3: win rate using ONLY prior matches (no leakage) ────────────
# shift(1) ensures the current match is excluded before expanding mean
long_wr = long.sort_values(["team", "Year"]).reset_index(drop=True)
long_wr["cum_win_rate"] = (
    long_wr.groupby("team")["win"]
           .transform(lambda x: x.shift(1).expanding().mean())
)
# Per (team, Year): take the last cumulative value = most recent snapshot before/at that year
win_rate_lookup = (
    long_wr.groupby(["team", "Year"])["cum_win_rate"]
           .last()
           .reset_index()
)

df = df.merge(
    win_rate_lookup.rename(columns={"team": "HomeTeamName",
                                    "cum_win_rate": "team1_win_rate"}),
    on=["HomeTeamName", "Year"], how="left"
)
df = df.merge(
    win_rate_lookup.rename(columns={"team": "AwayTeamName",
                                    "cum_win_rate": "team2_win_rate"}),
    on=["AwayTeamName", "Year"], how="left"
)
df["team1_win_rate"] = df["team1_win_rate"].fillna(0.5)
df["team2_win_rate"] = df["team2_win_rate"].fillna(0.5)

# ── Feature 4: head-to-head win rate using ONLY prior-year matches ────────────
# For each match row, scan only rows with Year < current Year
df_sorted = df.sort_values("Year").reset_index(drop=True)

h2h_rates = []
for idx, row in df_sorted.iterrows():
    home, away, year = row["HomeTeamName"], row["AwayTeamName"], row["Year"]
    prior = df_sorted[df_sorted["Year"] < year]
    # matches where this pair met in either direction
    mask = (
        ((prior["HomeTeamName"] == home) & (prior["AwayTeamName"] == away)) |
        ((prior["HomeTeamName"] == away) & (prior["AwayTeamName"] == home))
    )
    h2h_matches = prior[mask]
    if len(h2h_matches) == 0:
        h2h_rates.append(0.5)
    else:
        wins = sum(
            r["home_win"] if r["HomeTeamName"] == home else (1 - r["home_win"])
            for _, r in h2h_matches.iterrows()
        )
        h2h_rates.append(wins / len(h2h_matches))

df_sorted["head_to_head"] = h2h_rates
df = df_sorted.reset_index(drop=True)

# ── Feature 5: stage_num ──────────────────────────────────────────────────────
df["stage_num"] = df["Stage"].map(STAGE_MAP).fillna(0).astype(int)

# ── Features 6 & 7: rolling avg goals using ONLY prior matches ────────────────
# shift(1) already excludes the current match — this was already correct
long_sorted = long.sort_values(["team", "Year"]).reset_index(drop=True)
long_sorted["rolling_avg_goals"] = (
    long_sorted.groupby("team")["goals_scored"]
               .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
)

rolling_lookup = (
    long_sorted.sort_values(["team", "Year"])
               .groupby(["team", "Year"])["rolling_avg_goals"]
               .last()
               .reset_index()
)

df = df.merge(
    rolling_lookup.rename(columns={"team": "HomeTeamName",
                                    "rolling_avg_goals": "team1_avg_goals"}),
    on=["HomeTeamName", "Year"], how="left"
)
df = df.merge(
    rolling_lookup.rename(columns={"team": "AwayTeamName",
                                    "rolling_avg_goals": "team2_avg_goals"}),
    on=["AwayTeamName", "Year"], how="left"
)

overall_goal_avg = long["goals_scored"].mean()
df["team1_avg_goals"] = df["team1_avg_goals"].fillna(overall_goal_avg)
df["team2_avg_goals"] = df["team2_avg_goals"].fillna(overall_goal_avg)

# ── Feature 8: team1_is_host ──────────────────────────────────────────────────
df["team1_is_host"] = df.apply(
    lambda row: int(row["HomeTeamName"] in HOSTS.get(row["Year"], [])),
    axis=1
)

# ── Feature 9: coach_exp_diff (team1 coach WC appearances − team2) ────────────
df["team1_coach_exp"] = df["HomeTeamName"].map(COACH_WC_EXP).fillna(0).astype(int)
df["team2_coach_exp"] = df["AwayTeamName"].map(COACH_WC_EXP).fillna(0).astype(int)
df["coach_exp_diff"]  = df["team1_coach_exp"] - df["team2_coach_exp"]

# ── Features 10–12: recent international form (last 10 matches before WC) ────
print("Computing recent form from results.csv …")
res = pd.read_csv(RESULTS_PATH, parse_dates=["date"])

def team_recent_form(team_wc_name, wc_year, n=10, default=0.45):
    """
    Win rate of `team_wc_name` in the last `n` international matches
    played strictly before January 1 of `wc_year`.
    Draw counts as 0.5. Returns `default` if no history found.
    """
    name = RESULTS_NAME_MAP.get(team_wc_name, team_wc_name)
    cutoff = pd.Timestamp(f"{wc_year}-01-01")
    mask = (
        ((res["home_team"] == name) | (res["away_team"] == name)) &
        (res["date"] < cutoff)
    )
    past = res[mask].sort_values("date").tail(n)
    if past.empty:
        return default
    points = 0.0
    for _, r in past.iterrows():
        if r["home_team"] == name:
            if r["home_score"] > r["away_score"]:   points += 1.0
            elif r["home_score"] == r["away_score"]: points += 0.5
        else:
            if r["away_score"] > r["home_score"]:   points += 1.0
            elif r["away_score"] == r["home_score"]: points += 0.5
    return points / len(past)

df["team1_recent_form"] = [
    team_recent_form(row["HomeTeamName"], row["Year"])
    for _, row in df.iterrows()
]
df["team2_recent_form"] = [
    team_recent_form(row["AwayTeamName"], row["Year"])
    for _, row in df.iterrows()
]
df["form_diff"] = df["team1_recent_form"] - df["team2_recent_form"]

# Sample check
print("\nRecent form sample (5 rows):")
print(df[["Year","HomeTeamName","AwayTeamName",
          "team1_recent_form","team2_recent_form","form_diff"]].head().to_string())

# ── Features 13–17: confederation identity & strength ────────────────────────
_default_strength = 0.40   # fallback for unmapped confederations
df["team1_conf"]          = df["HomeTeamName"].map(CONFEDERATION_MAP).fillna("Unknown")
df["team2_conf"]          = df["AwayTeamName"].map(CONFEDERATION_MAP).fillna("Unknown")
df["team1_conf_strength"] = df["team1_conf"].map(CONF_STRENGTH).fillna(_default_strength)
df["team2_conf_strength"] = df["team2_conf"].map(CONF_STRENGTH).fillna(_default_strength)
df["conf_strength_diff"]  = df["team1_conf_strength"] - df["team2_conf_strength"]

# ── Assemble final feature dataframe ─────────────────────────────────────────
FEATURE_COLS = [
    "Year", "Stage", "stage_num",
    "HomeTeamName", "AwayTeamName",
    "HomeTeamGoals", "AwayTeamGoals",
    "home_win",
    "rank_diff",
    "team1_win_rate",
    "team2_win_rate",
    "head_to_head",
    "team1_avg_goals",
    "team2_avg_goals",
    "team1_is_host",
    "coach_exp_diff",
    "team1_conf",
    "team2_conf",
    "team1_conf_strength",
    "team2_conf_strength",
    "conf_strength_diff",
    "team1_recent_form",
    "team2_recent_form",
    "form_diff",
]

features_df = df[FEATURE_COLS].reset_index(drop=True)
features_df.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
print("\nFeature columns:")
for col in FEATURE_COLS:
    print(f"  {col}")
print(f"\nShape: {features_df.shape}")
print(f"\nFirst 5 rows:")
print(features_df.head().to_string())
print(f"\nNull counts:\n{features_df.isnull().sum()[features_df.isnull().sum() > 0]}")

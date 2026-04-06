import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA 2026 Win Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global theme CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a5c2e 0%, #0d3b1e 100%);
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 1.05rem; }
.main { background-color: #f4f9f4; }
.card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-left: 5px solid #1a5c2e;
}
.metric-box {
    background: #1a5c2e;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    text-align: center;
    color: #ffffff;
}
.metric-box .label { font-size: 0.85rem; opacity: 0.85; margin-bottom: 4px; }
.metric-box .value { font-size: 2rem; font-weight: 700; }
.stButton > button {
    background-color: #1a5c2e; color: white; border: none;
    border-radius: 8px; padding: 0.55rem 2rem;
    font-size: 1rem; font-weight: 600; width: 100%; transition: background 0.2s;
}
.stButton > button:hover { background-color: #22763a; }
th { background-color: #1a5c2e !important; color: white !important; }
h1, h2, h3 { color: #1a5c2e; }
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH      = os.path.join(ROOT, "models", "fifa_model.pkl")
SHAP_PATH       = os.path.join(ROOT, "models", "shap_importance.png")
FIX_PATH        = os.path.join(ROOT, "data", "fixtures_2026.csv")
FEAT_PATH       = os.path.join(ROOT, "data", "features.csv")
RESULTS_PATH    = os.path.join(ROOT, "data", "results.csv")
STAR_PATH       = os.path.join(ROOT, "data", "star_players_2026.csv")
WC_PATH         = os.path.join(ROOT, "data", "WorldCupMatches.csv")

STAR_PENALTY = 0.85   # win-prob multiplier when star player is absent

# ── Feature columns (must match trained model) ────────────────────────────────
FEATURE_COLS = [
    "rank_diff", "team1_win_rate", "team2_win_rate", "head_to_head",
    "stage_num", "team1_avg_goals", "team2_avg_goals", "team1_is_host",
    "coach_exp_diff",
    "team1_conf_strength", "team2_conf_strength", "conf_strength_diff",
    "team1_recent_form", "team2_recent_form", "form_diff",
]

RESULTS_NAME_MAP = {
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Serbia and Montenegro": "Serbia",
    "USA": "United States",
}

STAGE_OPTIONS  = ["Group Stage", "Round of 32", "Round of 16",
                  "Quarter-final", "Semi-final", "Final"]
STAGE_NUM_MAP  = {s: i for i, s in enumerate(STAGE_OPTIONS)}
HOSTS_2026     = {"USA", "Canada", "Mexico"}
DEFAULT_WIN_RATE  = 0.50
DEFAULT_AVG_GOALS = 1.35

CONFEDERATION_MAP = {
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
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Peru": "CONMEBOL", "Ecuador": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    "Japan": "AFC", "Korea Republic": "AFC", "South Korea": "AFC",
    "IR Iran": "AFC", "Saudi Arabia": "AFC", "Australia": "AFC",
    "China PR": "AFC", "Iraq": "AFC", "Qatar": "AFC", "Uzbekistan": "AFC",
    "Jordan": "AFC", "Indonesia": "AFC", "Korea DPR": "AFC",
    "Morocco": "CAF", "Senegal": "CAF", "Nigeria": "CAF", "Ghana": "CAF",
    "Cameroon": "CAF", "Egypt": "CAF", "Tunisia": "CAF", "Algeria": "CAF",
    "Côte d'Ivoire": "CAF", "Mali": "CAF", "South Africa": "CAF",
    "Angola": "CAF", "Togo": "CAF", "Cabo Verde": "CAF",
    "DR Congo": "CAF", "Tanzania": "CAF", "Comoros": "CAF", "Benin": "CAF",
    "USA": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Jamaica": "CONCACAF", "Panama": "CONCACAF",
    "Honduras": "CONCACAF", "El Salvador": "CONCACAF", "Haiti": "CONCACAF",
    "Curaçao": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "Guatemala": "CONCACAF", "Cuba": "CONCACAF",
    "New Zealand": "OFC",
}
CONF_STRENGTH = {
    "CONMEBOL": 0.58, "UEFA": 0.54, "CONCACAF": 0.38,
    "AFC": 0.35, "CAF": 0.34, "OFC": 0.28,
}
CONF_BADGE = {
    "UEFA":     ("#1565C0", "UEFA"),
    "CONMEBOL": ("#2E7D32", "CONMEBOL"),
    "AFC":      ("#C62828", "AFC"),
    "CAF":      ("#E65100", "CAF"),
    "CONCACAF": ("#6A1B9A", "CONCACAF"),
    "OFC":      ("#00695C", "OFC"),
}

COACH_WC_EXP = {
    "Argentina": 1, "Brazil": 0, "Uruguay": 1, "Colombia": 0,
    "Ecuador": 0, "Paraguay": 1, "Chile": 1, "Bolivia": 0, "Peru": 0, "Venezuela": 0,
    "France": 3, "Spain": 0, "Germany": 0, "England": 0, "Portugal": 1,
    "Netherlands": 1, "Croatia": 2, "Belgium": 0, "Switzerland": 0,
    "Serbia": 1, "Denmark": 0, "Austria": 0, "Turkey": 0, "Ukraine": 0,
    "Poland": 0, "Hungary": 0, "Slovakia": 0, "Slovenia": 0,
    "Scotland": 0, "Romania": 0, "Greece": 0, "Sweden": 0,
    "Norway": 0, "Bulgaria": 0, "Italy": 0, "Russia": 0,
    "Yugoslavia": 0, "Serbia and Montenegro": 0, "Bosnia and Herzegovina": 0,
    "Republic of Ireland": 0,
    "USA": 0, "Mexico": 2, "Canada": 0, "Costa Rica": 0, "Panama": 0,
    "Honduras": 0, "Jamaica": 0, "Haiti": 0, "Trinidad and Tobago": 0,
    "Japan": 1, "Korea Republic": 1, "South Korea": 1, "Australia": 0,
    "IR Iran": 0, "Saudi Arabia": 0, "Jordan": 0, "Uzbekistan": 0,
    "Korea DPR": 0, "China PR": 0, "Indonesia": 0, "Qatar": 0,
    "Morocco": 1, "Senegal": 1, "Nigeria": 0, "South Africa": 0,
    "Egypt": 0, "Algeria": 0, "Cameroon": 0, "Ghana": 0,
    "Côte d'Ivoire": 0, "Tunisia": 0, "Togo": 0, "Angola": 0,
    "Cabo Verde": 0, "Curaçao": 0, "New Zealand": 0,
}

ALL_2026_TEAMS = sorted([
    "Algeria", "Argentina", "Australia", "Austria", "Belgium", "Brazil",
    "Cabo Verde", "Canada", "Colombia", "Croatia", "Curaçao",
    "Côte d'Ivoire", "Ecuador", "Egypt", "England", "France", "Germany",
    "Ghana", "Haiti", "IR Iran", "Japan", "Jordan", "Mexico", "Morocco",
    "Netherlands", "New Zealand", "Norway", "Panama", "Paraguay",
    "Portugal", "Qatar", "Saudi Arabia", "Scotland", "Senegal",
    "South Africa", "South Korea", "Spain", "Switzerland", "Tunisia",
    "USA", "Uruguay", "Uzbekistan",
    "Winner UEFA Playoff A", "Winner UEFA Playoff B",
    "Winner UEFA Playoff C", "Winner UEFA Playoff D",
    "Winner FIFA Playoff 1", "Winner FIFA Playoff 2",
])

# ── Country flag emoji dictionary ─────────────────────────────────────────────
TEAM_FLAGS = {
    # CONMEBOL
    "Brazil": "🇧🇷", "Argentina": "🇦🇷", "Uruguay": "🇺🇾", "Colombia": "🇨🇴",
    "Ecuador": "🇪🇨", "Paraguay": "🇵🇾", "Peru": "🇵🇪", "Chile": "🇨🇱",
    "Bolivia": "🇧🇴", "Venezuela": "🇻🇪",
    # UEFA
    "France": "🇫🇷", "Spain": "🇪🇸", "Germany": "🇩🇪", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Portugal": "🇵🇹", "Netherlands": "🇳🇱", "Belgium": "🇧🇪", "Croatia": "🇭🇷",
    "Switzerland": "🇨🇭", "Serbia": "🇷🇸", "Denmark": "🇩🇰", "Austria": "🇦🇹",
    "Turkey": "🇹🇷", "Ukraine": "🇺🇦", "Poland": "🇵🇱", "Hungary": "🇭🇺",
    "Slovakia": "🇸🇰", "Slovenia": "🇸🇮", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Romania": "🇷🇴",
    "Greece": "🇬🇷", "Sweden": "🇸🇪", "Norway": "🇳🇴", "Bulgaria": "🇧🇬",
    "Italy": "🇮🇹", "Albania": "🇦🇱", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "Czech Republic": "🇨🇿", "Czechia": "🇨🇿",
    "Bosnia and Herzegovina": "🇧🇦",
    "Russia": "🇷🇺", "Yugoslavia": "🇾🇺",
    # AFC
    "Japan": "🇯🇵", "South Korea": "🇰🇷", "Korea Republic": "🇰🇷",
    "IR Iran": "🇮🇷", "Saudi Arabia": "🇸🇦", "Australia": "🇦🇺",
    "China PR": "🇨🇳", "Iraq": "🇮🇶", "Qatar": "🇶🇦", "Uzbekistan": "🇺🇿",
    "Jordan": "🇯🇴", "Indonesia": "🇮🇩", "Korea DPR": "🇰🇵",
    # CAF
    "Morocco": "🇲🇦", "Senegal": "🇸🇳", "Nigeria": "🇳🇬", "Ghana": "🇬🇭",
    "Cameroon": "🇨🇲", "Egypt": "🇪🇬", "Tunisia": "🇹🇳", "Algeria": "🇩🇿",
    "Côte d'Ivoire": "🇨🇮", "Mali": "🇲🇱", "South Africa": "🇿🇦",
    "Angola": "🇦🇴", "Togo": "🇹🇬", "Cabo Verde": "🇨🇻",
    "DR Congo": "🇨🇩", "Tanzania": "🇹🇿", "Comoros": "🇰🇲", "Benin": "🇧🇯",
    # CONCACAF
    "USA": "🇺🇸", "Mexico": "🇲🇽", "Canada": "🇨🇦", "Costa Rica": "🇨🇷",
    "Jamaica": "🇯🇲", "Panama": "🇵🇦", "Honduras": "🇭🇳", "El Salvador": "🇸🇻",
    "Haiti": "🇭🇹", "Curaçao": "🇨🇼", "Trinidad and Tobago": "🇹🇹",
    "Guatemala": "🇬🇹",
    # OFC
    "New Zealand": "🇳🇿",
}

def get_flag(team: str) -> str:
    """Return flag emoji for a team, falling back to a white flag."""
    return TEAM_FLAGS.get(team, "🏳️")

def flag_team(team: str) -> str:
    """Return '{flag} {team}' display string."""
    return f"{get_flag(team)} {team}"

# Pre-built display list and reverse lookup for selectboxes
_FLAGGED_TEAMS     = [flag_team(t) for t in ALL_2026_TEAMS]
_DISPLAY_TO_TEAM   = {flag_team(t): t for t in ALL_2026_TEAMS}

# ── Helper: confederation ─────────────────────────────────────────────────────
def get_conf(team):
    return CONFEDERATION_MAP.get(team, "Unknown")

def conf_badge_html(team):
    conf = get_conf(team)
    if conf in CONF_BADGE:
        bg, label = CONF_BADGE[conf]
        return (
            f'<span style="background:{bg};color:#fff;padding:2px 10px;'
            f'border-radius:12px;font-size:0.78rem;font-weight:600;'
            f'letter-spacing:0.03em;">{label}</span>'
        )
    return '<span style="background:#546E7A;color:#fff;padding:2px 10px;border-radius:12px;font-size:0.78rem;">—</span>'

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_features():
    return pd.read_csv(FEAT_PATH)

@st.cache_data
def load_fixtures():
    return pd.read_csv(FIX_PATH)

@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    return pd.read_csv(RESULTS_PATH, parse_dates=["date"])

@st.cache_data
def load_star_players():
    if not os.path.exists(STAR_PATH):
        return pd.DataFrame(columns=["team","star_player","position","club","availability"])
    return pd.read_csv(STAR_PATH)

# Normalise garbled / legacy names in WorldCupMatches.csv
_WC_CSV_NORM = {
    "Germany FR": "Germany", "German DR": "Germany",
    "Soviet Union": "Russia", "Dutch East Indies": "Indonesia",
    "Iran": "IR Iran",
    'rn">Bosnia and Herzegovina': "Bosnia and Herzegovina",
    'rn">Republic of Ireland':    "Republic of Ireland",
    'rn">Serbia and Montenegro':  "Serbia and Montenegro",
    'rn">Trinidad and Tobago':    "Trinidad and Tobago",
    'rn">United Arab Emirates':   "United Arab Emirates",
    "C\ufffdte d'Ivoire":          "Côte d'Ivoire",
}

@st.cache_data
def load_match_history():
    df = pd.read_csv(WC_PATH)
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    df = df.dropna(subset=["HomeTeamName","AwayTeamName","HomeTeamGoals","AwayTeamGoals"])
    df["HomeTeamGoals"] = df["HomeTeamGoals"].astype(int)
    df["AwayTeamGoals"] = df["AwayTeamGoals"].astype(int)
    df["Year"] = df["Year"].astype(int)
    df["HomeTeamName"] = df["HomeTeamName"].replace(_WC_CSV_NORM)
    df["AwayTeamName"] = df["AwayTeamName"].replace(_WC_CSV_NORM)
    return df

@st.cache_data
def build_team_stats():
    feat = load_features()
    home = feat[["HomeTeamName", "home_win", "team1_avg_goals"]].rename(
        columns={"HomeTeamName": "team", "home_win": "win", "team1_avg_goals": "goals"})
    away = feat[["AwayTeamName", "home_win", "team2_avg_goals"]].copy()
    away["win"] = 1 - away["home_win"]
    away = away[["AwayTeamName", "win", "team2_avg_goals"]].rename(
        columns={"AwayTeamName": "team", "team2_avg_goals": "goals"})
    combined = pd.concat([home, away])
    return (combined.groupby("team")
                    .agg(win_rate=("win","mean"), avg_goals=("goals","mean"))
                    .to_dict("index"))

@st.cache_data
def build_h2h():
    feat = load_features()
    h2h = {}
    for _, row in feat.iterrows():
        pair = (row["HomeTeamName"], row["AwayTeamName"])
        h2h.setdefault(pair, {"w": 0, "n": 0})
        h2h[pair]["n"] += 1
        h2h[pair]["w"] += row["home_win"]
    return {k: v["w"] / v["n"] for k, v in h2h.items()}

# ── Star player helpers ───────────────────────────────────────────────────────
def get_star_player(team):
    """Return dict {star_player, position, club, availability} or None."""
    sp = load_star_players()
    row = sp[sp["team"] == team]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

def star_card_html(team, star, injured=False):
    """Render a styled star player info badge."""
    pos_colors = {
        "Forward": "#e65100", "Midfielder": "#1565c0",
        "Defender": "#2e7d32", "Goalkeeper": "#6a1b9a",
    }
    pos_bg = pos_colors.get(star.get("position", ""), "#555")
    status_bg = "#c62828" if injured else "#2e7d32"
    status_txt = "INJURED / SUSPENDED" if injured else "AVAILABLE"
    return (
        f'<div style="background:#f8f9fa;border-radius:10px;padding:10px 14px;'
        f'border-left:4px solid {status_bg};margin-top:6px;">'
        f'<div style="font-size:0.82rem;color:#555;margin-bottom:2px;">⭐ Star Player</div>'
        f'<div style="font-size:1.05rem;font-weight:700;color:#1a1a1a;">'
        f'{star.get("star_player","—")}</div>'
        f'<div style="margin-top:4px;">'
        f'<span style="background:{pos_bg};color:#fff;padding:1px 8px;'
        f'border-radius:8px;font-size:0.72rem;font-weight:600;margin-right:6px;">'
        f'{star.get("position","")}</span>'
        f'<span style="font-size:0.78rem;color:#555;">{star.get("club","")}</span>'
        f'</div>'
        f'<div style="margin-top:6px;">'
        f'<span style="background:{status_bg};color:#fff;padding:1px 8px;'
        f'border-radius:8px;font-size:0.72rem;font-weight:600;">{status_txt}</span>'
        f'</div></div>'
    )

def apply_star_penalty(p1, p2, star1_injured, star2_injured):
    """Apply 0.85 multiplier when a star player is absent, then renormalise."""
    raw1 = p1 * (STAR_PENALTY if star1_injured else 1.0)
    raw2 = p2 * (STAR_PENALTY if star2_injured else 1.0)
    total = raw1 + raw2
    if total == 0:
        return p1, p2
    return round(raw1 / total * 100, 1), round(raw2 / total * 100, 1)

# ── Recent form helpers ───────────────────────────────────────────────────────
def _results_name(team):
    return RESULTS_NAME_MAP.get(team, team)

def get_recent_form(team, before_year=2026, n=10, default=0.45):
    res = load_results()
    if res is None:
        return default
    name = _results_name(team)
    cutoff = pd.Timestamp(f"{before_year}-01-01")
    mask = ((res["home_team"] == name) | (res["away_team"] == name)) & (res["date"] < cutoff)
    past = res[mask].sort_values("date").tail(n)
    if past.empty:
        return default
    pts = 0.0
    for _, r in past.iterrows():
        if r["home_team"] == name:
            if r["home_score"] > r["away_score"]:    pts += 1.0
            elif r["home_score"] == r["away_score"]: pts += 0.5
        else:
            if r["away_score"] > r["home_score"]:    pts += 1.0
            elif r["away_score"] == r["home_score"]: pts += 0.5
    return pts / len(past)

def get_last5_results(team, before_year=2026):
    res = load_results()
    if res is None:
        return []
    name = _results_name(team)
    cutoff = pd.Timestamp(f"{before_year}-01-01")
    mask = ((res["home_team"] == name) | (res["away_team"] == name)) & (res["date"] < cutoff)
    past = res[mask].sort_values("date").tail(5)
    outcomes = []
    for _, r in past.iterrows():
        if r["home_team"] == name:
            gs, gc = r["home_score"], r["away_score"]
        else:
            gs, gc = r["away_score"], r["home_score"]
        if gs > gc:   outcomes.append("W")
        elif gs < gc: outcomes.append("L")
        else:         outcomes.append("D")
    return outcomes

def form_trend(team, before_year=2026):
    res = load_results()
    if res is None:
        return "→", "#888"
    name = _results_name(team)
    cutoff = pd.Timestamp(f"{before_year}-01-01")
    mask = ((res["home_team"] == name) | (res["away_team"] == name)) & (res["date"] < cutoff)
    past = res[mask].sort_values("date").tail(6)
    if len(past) < 4:
        return "→", "#888"
    def win_pts(rows):
        pts = 0.0
        for _, r in rows.iterrows():
            if r["home_team"] == name:
                gs, gc = r["home_score"], r["away_score"]
            else:
                gs, gc = r["away_score"], r["home_score"]
            if gs > gc:    pts += 1.0
            elif gs == gc: pts += 0.5
        return pts / len(rows)
    if win_pts(past.iloc[-3:]) > win_pts(past.iloc[:3]) + 0.15: return "↑", "#2e7d32"
    if win_pts(past.iloc[-3:]) < win_pts(past.iloc[:3]) - 0.15: return "↓", "#c62828"
    return "→", "#888888"

def form_dots_html(outcomes):
    colors = {"W": "#2e7d32", "D": "#888888", "L": "#c62828"}
    dots = [
        f'<span title="{o}" style="display:inline-block;width:28px;height:28px;'
        f'border-radius:50%;background:{colors.get(o,"#888")};color:#fff;'
        f'font-size:0.7rem;font-weight:700;text-align:center;line-height:28px;margin:0 3px;">{o}</span>'
        for o in outcomes
    ]
    return "".join(dots) if dots else '<span style="color:#888;font-size:0.85rem;">No data</span>'

# ── H2H helpers ──────────────────────────────────────────────────────────────

# Maps app dropdown names → WorldCupMatches.csv names (only where they differ)
TEAM_NAME_MAP = {
    "South Korea": "Korea Republic",
    "North Korea":  "Korea DPR",
    "DR Congo":     "Congo DR",
}

def normalize_team_name(team, df_teams):
    """Return the best matching name found in df_teams set."""
    if team in df_teams:
        return team
    mapped = TEAM_NAME_MAP.get(team, team)
    if mapped in df_teams:
        return mapped
    return team

@st.cache_data
def get_h2h_stats(team1, team2, mdf):
    """Return H2H stats dict or None if no matches exist."""
    all_teams = set(mdf["HomeTeamName"]) | set(mdf["AwayTeamName"])
    n1 = normalize_team_name(team1, all_teams)
    n2 = normalize_team_name(team2, all_teams)
    h2h = mdf[
        ((mdf["HomeTeamName"] == n1) & (mdf["AwayTeamName"] == n2)) |
        ((mdf["HomeTeamName"] == n2) & (mdf["AwayTeamName"] == n1))
    ].copy()
    if len(h2h) == 0:
        return None
    h2h = h2h.sort_values("Year", ascending=False)
    stats = {"total": len(h2h), "team1_wins": 0, "team2_wins": 0, "draws": 0,
             "last_5": [], "all_matches": []}
    for _, row in h2h.iterrows():
        home  = row["HomeTeamName"]
        hg, ag = row["HomeTeamGoals"], row["AwayTeamGoals"]
        year  = int(row["Year"])
        stage = str(row.get("Stage", "")).strip()
        if home == n1:
            score = f"{hg} – {ag}"
            if   hg > ag: result = "W"; stats["team1_wins"] += 1
            elif hg < ag: result = "L"; stats["team2_wins"] += 1
            else:          result = "D"; stats["draws"]     += 1
        else:
            score = f"{ag} – {hg}"
            if   ag > hg: result = "W"; stats["team1_wins"] += 1
            elif ag < hg: result = "L"; stats["team2_wins"] += 1
            else:          result = "D"; stats["draws"]     += 1
        stats["all_matches"].append({
            "year": year, "stage": stage, "score": score,
            "result": result, "home": home,
            "home_goals": hg if home == n1 else ag,
            "away_goals": ag if home == n1 else hg,
        })
    stats["last_5"] = stats["all_matches"][:5]
    return stats

def h2h_win_bar(team1, team2, stats):
    """Plotly stacked horizontal bar showing win %."""
    total = stats["total"]
    pct1  = round(stats["team1_wins"] / total * 100, 1)
    pctd  = round(stats["draws"]      / total * 100, 1)
    pct2  = round(stats["team2_wins"] / total * 100, 1)
    f1, f2 = get_flag(team1), get_flag(team2)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[pct1], y=[""], orientation="h", name=team1,
        marker_color="#2e7d32",
        text=[f"{f1} {team1}  {pct1}%"],
        textposition="inside", insidetextanchor="middle",
    ))
    fig.add_trace(go.Bar(
        x=[pctd], y=[""], orientation="h", name="Draw",
        marker_color="#9e9e9e",
        text=[f"{pctd}% Draw" if pctd >= 10 else ""],
        textposition="inside", insidetextanchor="middle",
    ))
    fig.add_trace(go.Bar(
        x=[pct2], y=[""], orientation="h", name=team2,
        marker_color="#c62828",
        text=[f"{pct2}%  {team2} {f2}"],
        textposition="inside", insidetextanchor="middle",
    ))
    fig.update_layout(
        barmode="stack", height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def h2h_dots_html(matches):
    """Colored circles for up to last 10 WC meetings (from team1 perspective)."""
    color_map = {"W": "#2e7d32", "D": "#9e9e9e", "L": "#c62828"}
    parts = []
    for m in matches[:10]:
        c = color_map.get(m["result"], "#888")
        parts.append(
            f'<span title="{m["year"]}: {m["score"]}" '
            f'style="display:inline-block;width:24px;height:24px;border-radius:50%;'
            f'background:{c};margin:2px;cursor:default;"></span>'
        )
    return "".join(parts) if parts else '<span style="color:#888;font-size:0.85rem;">—</span>'

def h2h_goals_chart(team1, team2, stats):
    """Plotly grouped bar: goals per year."""
    years  = [str(m["year"]) for m in reversed(stats["all_matches"])]
    g1     = [m["home_goals"]  for m in reversed(stats["all_matches"])]
    g2     = [m["away_goals"]  for m in reversed(stats["all_matches"])]
    stages = [m["stage"]       for m in reversed(stats["all_matches"])]
    f1, f2 = get_flag(team1), get_flag(team2)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"{f1} {team1}", x=years, y=g1,
        marker_color="#1565c0",
        customdata=stages, hovertemplate="%{x} — %{customdata}<br>Goals: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name=f"{f2} {team2}", x=years, y=g2,
        marker_color="#c62828",
        customdata=stages, hovertemplate="%{x} — %{customdata}<br>Goals: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title="Goals Scored in Each Meeting",
        barmode="group", height=280,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#f4f9f4", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", tickangle=-30),
        yaxis=dict(title="Goals", dtick=1),
    )
    return fig

def show_h2h_section(team1, team2, mdf, compact=False):
    """Render the full H2H section. compact=True shows fewer rows (for fixtures page)."""
    h2h = get_h2h_stats(team1, team2, mdf)
    f1, f2 = get_flag(team1), get_flag(team2)

    if h2h is None:
        st.info(f"No World Cup history between {f1} {team1} and {f2} {team2}.")
        return None

    # ── Element 1 — Summary metrics ───────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(
            f'<div style="text-align:center;background:#f4f9f4;border-radius:10px;'
            f'padding:12px 8px;border:1px solid #ddd;">'
            f'<div style="font-size:0.8rem;color:#666;">Total Meetings</div>'
            f'<div style="font-size:2rem;font-weight:700;color:#1a5c2e;">{h2h["total"]}</div>'
            f'</div>', unsafe_allow_html=True)
    with mc2:
        st.markdown(
            f'<div style="text-align:center;background:#f4f9f4;border-radius:10px;'
            f'padding:12px 8px;border:1px solid #ddd;">'
            f'<div style="font-size:0.8rem;color:#666;">{f1} {team1} Wins</div>'
            f'<div style="font-size:2rem;font-weight:700;color:#2e7d32;">{h2h["team1_wins"]}</div>'
            f'</div>', unsafe_allow_html=True)
    with mc3:
        st.markdown(
            f'<div style="text-align:center;background:#f4f9f4;border-radius:10px;'
            f'padding:12px 8px;border:1px solid #ddd;">'
            f'<div style="font-size:0.8rem;color:#666;">Draws</div>'
            f'<div style="font-size:2rem;font-weight:700;color:#757575;">{h2h["draws"]}</div>'
            f'</div>', unsafe_allow_html=True)
    with mc4:
        st.markdown(
            f'<div style="text-align:center;background:#f4f9f4;border-radius:10px;'
            f'padding:12px 8px;border:1px solid #ddd;">'
            f'<div style="font-size:0.8rem;color:#666;">{f2} {team2} Wins</div>'
            f'<div style="font-size:2rem;font-weight:700;color:#c62828;">{h2h["team2_wins"]}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Element 2 — Win percentage bar ────────────────────────────────────────
    st.plotly_chart(h2h_win_bar(team1, team2, h2h), use_container_width=True,
                    config={"displayModeBar": False})

    # ── Element 6 — Animated result dots (last 10) ────────────────────────────
    st.markdown(
        f'<div style="font-size:0.82rem;color:#555;margin-bottom:4px;">'
        f'Last {min(len(h2h["all_matches"]),10)} WC meetings — '
        f'<span style="color:#2e7d32;font-weight:600;">● Win</span>&nbsp;'
        f'<span style="color:#9e9e9e;font-weight:600;">● Draw</span>&nbsp;'
        f'<span style="color:#c62828;font-weight:600;">● Loss</span>'
        f'&nbsp;(from {f1} {team1}\'s perspective)</div>',
        unsafe_allow_html=True)
    st.markdown(h2h_dots_html(h2h["all_matches"]), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Element 3 — Last matches timeline ─────────────────────────────────────
    limit = 3 if compact else 5
    st.markdown(f"**Last {min(limit, len(h2h['all_matches']))} World Cup Meetings**")
    rc_map = {"W": ("#2e7d32", "WIN"), "D": ("#757575", "DRAW"), "L": ("#c62828", "LOSS")}
    for m in h2h["last_5"][:limit]:
        rc, rl = rc_map.get(m["result"], ("#555", m["result"]))
        mc_a, mc_b, mc_c = st.columns([2, 4, 1])
        with mc_a:
            st.markdown(
                f'<div style="font-size:0.82rem;color:#555;padding-top:4px;">'
                f'<b>{m["year"]}</b><br>'
                f'<span style="font-size:0.75rem;">{m["stage"]}</span></div>',
                unsafe_allow_html=True)
        with mc_b:
            st.markdown(
                f'<div style="text-align:center;font-size:1rem;font-weight:600;padding-top:2px;">'
                f'{f1} {team1}&nbsp;&nbsp;'
                f'<span style="background:#1a5c2e;color:#fff;padding:2px 12px;border-radius:6px;">'
                f'{m["score"]}</span>'
                f'&nbsp;&nbsp;{f2} {team2}</div>',
                unsafe_allow_html=True)
        with mc_c:
            st.markdown(
                f'<div style="text-align:center;padding-top:2px;">'
                f'<span style="background:{rc};color:#fff;padding:2px 10px;border-radius:8px;'
                f'font-size:0.78rem;font-weight:700;">{rl}</span></div>',
                unsafe_allow_html=True)

    if not compact:
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Element 4 — Goals chart ────────────────────────────────────────────
        st.plotly_chart(h2h_goals_chart(team1, team2, h2h), use_container_width=True,
                        config={"displayModeBar": False})

        # ── Element 5 — Fun facts ──────────────────────────────────────────────
        last    = h2h["all_matches"][0]
        big_win = max(h2h["all_matches"],
                      key=lambda m: abs(m["home_goals"] - m["away_goals"]))
        big_diff = abs(big_win["home_goals"] - big_win["away_goals"])
        total_goals_all = sum(m["home_goals"] + m["away_goals"]
                              for m in h2h["all_matches"])
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown(
                f'<div style="background:#f4f9f4;border-radius:8px;padding:10px 14px;'
                f'border:1px solid #ddd;font-size:0.84rem;">'
                f'<div style="color:#888;margin-bottom:3px;">Last Meeting</div>'
                f'<b>{last["year"]}</b> — {last["score"]}</div>',
                unsafe_allow_html=True)
        with fc2:
            st.markdown(
                f'<div style="background:#f4f9f4;border-radius:8px;padding:10px 14px;'
                f'border:1px solid #ddd;font-size:0.84rem;">'
                f'<div style="color:#888;margin-bottom:3px;">Biggest Win</div>'
                f'<b>{big_win["year"]}</b> — {big_win["score"]} ({big_diff}-goal diff)</div>',
                unsafe_allow_html=True)
        with fc3:
            st.markdown(
                f'<div style="background:#f4f9f4;border-radius:8px;padding:10px 14px;'
                f'border:1px solid #ddd;font-size:0.84rem;">'
                f'<div style="color:#888;margin-bottom:3px;">Total Goals</div>'
                f'<b>{total_goals_all}</b> goals across {h2h["total"]} meeting'
                f'{"s" if h2h["total"] > 1 else ""}</div>',
                unsafe_allow_html=True)

    return h2h

# ── Feature row builder ───────────────────────────────────────────────────────
def get_win_rate(team, stats):  return stats.get(team, {}).get("win_rate",  DEFAULT_WIN_RATE)
def get_avg_goals(team, stats): return stats.get(team, {}).get("avg_goals", DEFAULT_AVG_GOALS)

def get_h2h(team1, team2, h2h_dict):
    if (team1, team2) in h2h_dict: return h2h_dict[(team1, team2)]
    if (team2, team1) in h2h_dict: return 1 - h2h_dict[(team2, team1)]
    return 0.5

def build_feature_row(team1, team2, rank1, rank2, stage_num, stats, h2h_dict, year=2026):
    c1, c2  = get_conf(team1), get_conf(team2)
    cs1, cs2 = CONF_STRENGTH.get(c1, 0.40), CONF_STRENGTH.get(c2, 0.40)
    ce1, ce2 = COACH_WC_EXP.get(team1, 0), COACH_WC_EXP.get(team2, 0)
    rf1, rf2 = get_recent_form(team1, before_year=year), get_recent_form(team2, before_year=year)
    return pd.DataFrame([{
        "rank_diff":           rank1 - rank2,
        "team1_win_rate":      get_win_rate(team1, stats),
        "team2_win_rate":      get_win_rate(team2, stats),
        "head_to_head":        get_h2h(team1, team2, h2h_dict),
        "stage_num":           stage_num,
        "team1_avg_goals":     get_avg_goals(team1, stats),
        "team2_avg_goals":     get_avg_goals(team2, stats),
        "team1_is_host":       int(team1 in HOSTS_2026),
        "coach_exp_diff":      ce1 - ce2,
        "team1_conf_strength": cs1,
        "team2_conf_strength": cs2,
        "conf_strength_diff":  cs1 - cs2,
        "team1_recent_form":   rf1,
        "team2_recent_form":   rf2,
        "form_diff":           rf1 - rf2,
    }])

def predict_prob(mdl, X_row):
    p1 = float(mdl.predict_proba(X_row)[0][1])
    return round(p1 * 100, 1), round((1 - p1) * 100, 1)

# ── Charts ────────────────────────────────────────────────────────────────────
def make_gauge(prob, team_name):
    color = "#1a5c2e" if prob >= 50 else "#c0392b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        title={"text": f"<b>{team_name}</b><br>Win Probability",
               "font": {"size": 16, "color": "#1a5c2e"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555",
                     "tickfont": {"size": 11}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#f0f0f0", "borderwidth": 0,
            "steps": [{"range": [0,  40], "color": "#fdecea"},
                      {"range": [40, 60], "color": "#fff9e6"},
                      {"range": [60,100], "color": "#e8f5e9"}],
            "threshold": {"line": {"color": "#1a5c2e", "width": 3},
                          "thickness": 0.8, "value": 50},
        },
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def get_conf_color(team):
    """Return hex colour for a team's confederation (for bar charts)."""
    return CONF_BADGE.get(get_conf(team), ("#546E7A",""))[0]

def show_shap_section():
    st.markdown("#### SHAP Feature Importance (XGBoost)")
    if os.path.exists(SHAP_PATH):
        st.image(Image.open(SHAP_PATH), use_container_width=True)
    else:
        st.info("SHAP importance plot not found. Run `src/model.py` first.")

# ═══════════════════════════════════════════════════════════════════════════════
# Tournament Bracket — constants & simulation engine
# ═══════════════════════════════════════════════════════════════════════════════

# Normalize group team names → app names
_GRP_NORM = {
    "Ivory Coast": "Côte d'Ivoire",
    "Curacao":     "Curaçao",
    "Czech Republic": "Czech Republic",   # same in CONFEDERATION_MAP
}
def _gn(t): return _GRP_NORM.get(t, t)

GROUPS = {
    "A": [_gn(t) for t in ["Mexico",    "South Korea", "Czech Republic", "South Africa"]],
    "B": [_gn(t) for t in ["Switzerland","Canada",     "Bosnia and Herzegovina", "Qatar"]],
    "C": [_gn(t) for t in ["Brazil",    "Morocco",    "Scotland",    "Haiti"]],
    "D": [_gn(t) for t in ["Uruguay",   "Germany",    "Ivory Coast", "DR Congo"]],
    "E": [_gn(t) for t in ["USA",       "England",    "Panama",      "Algeria"]],
    "F": [_gn(t) for t in ["Portugal",  "Argentina",  "Iraq",        "Indonesia"]],
    "G": [_gn(t) for t in ["France",    "Japan",      "Peru",        "Tanzania"]],
    "H": [_gn(t) for t in ["Spain",     "Netherlands","Jamaica",     "Benin"]],
    "I": [_gn(t) for t in ["Belgium",   "Croatia",    "Chile",       "Comoros"]],
    "J": [_gn(t) for t in ["Colombia",  "Ecuador",    "Jordan",      "Curacao"]],
    "K": [_gn(t) for t in ["Nigeria",   "Egypt",      "Venezuela",   "El Salvador"]],
    "L": [_gn(t) for t in ["Australia", "Senegal",    "Honduras",    "Uzbekistan"]],
}

# Approximate 2026 FIFA rankings for all tournament teams
TEAM_RANK = {
    "Argentina": 1,  "France": 2,    "England": 4,   "Belgium": 3,
    "Brazil": 5,     "Portugal": 6,  "Netherlands": 7,"Spain": 8,
    "Colombia": 9,   "Germany": 12,  "USA": 11,      "Croatia": 10,
    "Uruguay": 14,   "Mexico": 15,   "Morocco": 13,  "Japan": 18,
    "Senegal": 20,   "Switzerland": 19, "Denmark": 21,"South Korea": 22,
    "Australia": 24,"IR Iran": 25,  "Ecuador": 23,  "Canada": 17,
    "Scotland": 30,  "Nigeria": 28,  "Egypt": 32,    "Turkey": 29,
    "Qatar": 35,     "South Africa": 54,"Serbia": 26, "Austria": 27,
    "Ukraine": 22,   "Czech Republic": 40, "Jamaica": 58, "Panama": 60,
    "Algeria": 34,   "Tunisia": 36,  "Cameroon": 43, "Ghana": 60,
    "Côte d'Ivoire": 48, "Mali": 55, "DR Congo": 63, "Benin": 95,
    "Tanzania": 122, "Comoros": 100, "Haiti": 85,    "Honduras": 72,
    "El Salvador": 76,"Indonesia": 134,"Iraq": 68,   "Jordan": 87,
    "Curaçao": 90,   "Bolivia": 80,  "Venezuela": 52,"Peru": 37,
    "Chile": 33,     "Paraguay": 50, "Cabo Verde": 78,"New Zealand": 102,
    "Bosnia and Herzegovina": 67, "Uzbekistan": 74,
    "Saudi Arabia": 56, "Norway": 31,
}

ALL_BRACKET_TEAMS = [t for grp in GROUPS.values() for t in grp]

@st.cache_data
def precompute_base_probs():
    """Compute base win-prob matrix for all pairs of tournament teams."""
    mdl = load_model()
    sts = build_team_stats()
    h2h = build_h2h()
    probs = {}
    teams = ALL_BRACKET_TEAMS
    for i, t1 in enumerate(teams):
        for t2 in teams[i+1:]:
            r1 = TEAM_RANK.get(t1, 35)
            r2 = TEAM_RANK.get(t2, 35)
            X  = build_feature_row(t1, t2, r1, r2, 0, sts, h2h)
            p  = float(mdl.predict_proba(X)[0][1])
            probs[(t1, t2)] = p
            probs[(t2, t1)] = 1.0 - p
    return probs

def _sim_match(t1, t2, base_probs, upset, rng, stage=0,
               star_on=False, inj=None):
    """Return (winner, loser, prob_t1_wins_after_noise)."""
    p = base_probs.get((t1, t2), 0.5)
    if star_on and inj:
        raw1 = p * 100 * (STAR_PENALTY if t1 in inj else 1.0)
        raw2 = (1-p) * 100 * (STAR_PENALTY if t2 in inj else 1.0)
        tot  = raw1 + raw2
        p    = (raw1 / tot) if tot else p
    if upset > 0:
        p = float(np.clip(p + rng.uniform(-upset * 0.5, upset * 0.5), 0.05, 0.95))
    winner, loser = (t1, t2) if p >= 0.5 else (t2, t1)
    return winner, loser, p

def _sim_group(teams, base_probs, upset, rng, star_on, inj):
    """Simulate a 4-team group; return sorted standings list."""
    pts   = {t: 0   for t in teams}
    wpsum = {t: 0.0 for t in teams}
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            t1, t2 = teams[i], teams[j]
            p = base_probs.get((t1, t2), 0.5)
            if star_on and inj:
                raw1 = p * 100 * (STAR_PENALTY if t1 in inj else 1.0)
                raw2 = (1-p) * 100 * (STAR_PENALTY if t2 in inj else 1.0)
                tot  = raw1 + raw2; p = (raw1/tot) if tot else p
            if upset > 0:
                p = float(np.clip(p + rng.uniform(-upset*0.5, upset*0.5), 0.05, 0.95))
            if p > 0.55:
                pts[t1] += 3; wpsum[t1] += p
            elif p < 0.45:
                pts[t2] += 3; wpsum[t2] += (1-p)
            else:
                pts[t1] += 1; pts[t2] += 1
                wpsum[t1] += p; wpsum[t2] += (1-p)
    return sorted(teams, key=lambda t: (pts[t], wpsum[t]), reverse=True), pts, wpsum

def simulate_tournament(upset, star_on, inj_set, seed, base_probs):
    """
    Full tournament simulation.
    Returns dict with group_standings, bracket_rounds, champion, path_to_glory.
    """
    rng = np.random.default_rng(seed)

    # ── Group stage ───────────────────────────────────────────────────────────
    group_standings = {}
    all_third = []
    r32_pairs  = []    # (t1, t2) from group cross-pairing

    group_keys = list(GROUPS.keys())   # A..L
    for gk in group_keys:
        teams = GROUPS[gk]
        standing, pts, wps = _sim_group(teams, base_probs, upset, rng, star_on, inj_set)
        group_standings[gk] = [
            {"team": t, "pts": pts[t], "wps": wps[t],
             "rank": i+1, "group": gk}
            for i, t in enumerate(standing)
        ]
        all_third.append(group_standings[gk][2])   # 3rd-place entry

    # ── Best 8 third-place teams ──────────────────────────────────────────────
    best8_3rd = sorted(all_third,
                       key=lambda x: (x["pts"], x["wps"]), reverse=True)[:8]
    best8_3rd_teams = [x["team"] for x in best8_3rd]

    # ── Build R32 bracket (paired groups) ─────────────────────────────────────
    pair_order = [("A","B"),("C","D"),("E","F"),("G","H"),("I","J"),("K","L")]
    r32 = []
    for g1, g2 in pair_order:
        s1, s2 = group_standings[g1], group_standings[g2]
        r32.append((s1[0]["team"], s2[1]["team"]))   # 1st vs 2nd (cross)
        r32.append((s2[0]["team"], s1[1]["team"]))   # 1st vs 2nd (cross)
    # Best-8 3rd place matches (4 matches)
    for k in range(4):
        r32.append((best8_3rd_teams[k], best8_3rd_teams[7-k]))

    # ── Simulate knockout rounds ──────────────────────────────────────────────
    rounds = {"R32": [], "R16": [], "QF": [], "SF": [], "Final": []}
    stage_num = {"R32": 1, "R16": 2, "QF": 3, "SF": 4, "Final": 5}

    current_matches = r32
    for rname in ["R32", "R16", "QF", "SF", "Final"]:
        winners = []
        for t1, t2 in current_matches:
            w, l, p = _sim_match(t1, t2, base_probs, upset, rng,
                                  stage=stage_num[rname],
                                  star_on=star_on, inj=inj_set)
            wp = p if w == t1 else 1 - p
            rounds[rname].append({
                "t1": t1, "t2": t2, "winner": w, "loser": l,
                "p_t1": round(p*100, 1), "p_winner": round(wp*100, 1)
            })
            winners.append(w)
        # Pair winners for next round
        current_matches = [(winners[i], winners[i+1])
                           for i in range(0, len(winners)-1, 2)]

    champion = rounds["Final"][0]["winner"]

    # ── Path to glory ─────────────────────────────────────────────────────────
    path = []
    for rname, matches in rounds.items():
        for m in matches:
            if m["winner"] == champion:
                opp = m["t2"] if m["t1"] == champion else m["t1"]
                path.append({"round": rname, "opponent": opp,
                              "win_prob": m["p_winner"]})

    return {"group_standings": group_standings,
            "rounds": rounds,
            "champion": champion,
            "path": path}

def run_n_simulations(n, upset, star_on, inj_set, base_probs):
    """Run n tournaments, return champion_counts and final_pairs."""
    champ_counts = {}
    final_pairs  = {}
    for seed in range(n):
        res = simulate_tournament(upset, star_on, inj_set, seed, base_probs)
        champ = res["champion"]
        champ_counts[champ] = champ_counts.get(champ, 0) + 1
        final = res["rounds"]["Final"][0]
        pair  = tuple(sorted([final["t1"], final["t2"]]))
        final_pairs[pair] = final_pairs.get(pair, 0) + 1
    return champ_counts, final_pairs

# ── Bracket match card HTML ────────────────────────────────────────────────────
def match_card(m, compact=False):
    """Return HTML for one bracket match."""
    t1, t2  = m["t1"], m["t2"]
    w       = m["winner"]
    p1      = m["p_t1"]
    p2      = round(100 - p1, 1)
    wstyle  = "border-left:3px solid #2e7d32;background:#f0faf0;"
    lstyle  = "border-left:3px solid #ccc;background:#f9f9f9;"
    s1 = wstyle if w == t1 else lstyle
    s2 = wstyle if w == t2 else lstyle
    d  = f'<div style="font-size:0.72rem;margin-bottom:6px;color:#888;">{m.get("label","")}</div>' if not compact else ""
    return (
        f'<div style="border:1px solid #e0e0e0;border-radius:8px;'
        f'overflow:hidden;margin-bottom:8px;font-size:0.85rem;">'
        f'{d}'
        f'<div style="{s1}padding:5px 10px;">'
        f'{flag_team(t1)}'
        f'<span style="float:right;color:#555;font-size:0.78rem;">{p1}%</span></div>'
        f'<div style="{s2}padding:5px 10px;">'
        f'{flag_team(t2)}'
        f'<span style="float:right;color:#555;font-size:0.78rem;">{p2}%</span></div>'
        f'</div>'
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚽ FIFA 2026")
    st.markdown("### Win Probability Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Match Simulator", "2026 Fixture Predictions",
         "Tournament Bracket", "Star Players 2026"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    # Bracket-specific controls
    if page == "Tournament Bracket":
        st.markdown("#### Simulator Settings")
        upset_factor = st.slider("Upset Factor", 0.0, 1.0, 0.10, 0.05,
                                 help="0 = model always wins · 1 = random")
        star_impact  = st.toggle("Star Player Impact", value=True,
                                 help="Apply 15% penalty for injured stars")
        sim_seed     = st.number_input("Seed", value=42, step=1,
                                       help="Fix for reproducible results")
        st.markdown("---")
    else:
        upset_factor, star_impact, sim_seed = 0.10, True, 42
    st.caption("Model: Logistic Regression\nData: World Cup 1994–2022 + FIFA Rankings")

# ── Shared resources (cached — only computed once, returned instantly on rerun)
@st.cache_resource
def get_shared_resources():
    return load_model(), build_team_stats(), build_h2h()

model, stats, h2h_dict = get_shared_resources()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Match Simulator
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Match Simulator":
    st.markdown("# ⚽ FIFA World Cup 2026 Win Probability Predictor")
    st.markdown("🏆 Predict match outcomes using ML + historical data")
    st.markdown("---")

    # ── Match Setup ───────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Match Setup")
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        team1_display = st.selectbox("🟢 Team 1 (Home)", _FLAGGED_TEAMS,
                                     index=_FLAGGED_TEAMS.index(flag_team("Brazil")))
        team1 = _DISPLAY_TO_TEAM[team1_display]
        st.markdown(conf_badge_html(team1), unsafe_allow_html=True)
        rank1 = st.slider("Team 1 FIFA Ranking", 1, 50, 5)
    with col2:
        team2_display = st.selectbox("⚪ Team 2 (Away)", _FLAGGED_TEAMS,
                                     index=_FLAGGED_TEAMS.index(flag_team("France")))
        team2 = _DISPLAY_TO_TEAM[team2_display]
        st.markdown(conf_badge_html(team2), unsafe_allow_html=True)
        rank2 = st.slider("Team 2 FIFA Ranking", 1, 50, 3)
    with col3:
        stage = st.selectbox("🏆 Match Stage", STAGE_OPTIONS, index=0)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict Win Probability")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Star Player Availability ───────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⭐ Star Player Availability")
    sp_col1, sp_col2 = st.columns(2)
    star1 = get_star_player(team1)
    star2 = get_star_player(team2)

    with sp_col1:
        if star1:
            star1_injured = not st.toggle(
                f"Available — {star1['star_player']}", value=True, key="sp1")
            st.markdown(star_card_html(team1, star1, star1_injured), unsafe_allow_html=True)
            if star1_injured:
                st.warning(f"⚠️ Playing without **{star1['star_player']}**")
        else:
            star1_injured = False
            st.caption(f"No star player data for {team1}")

    with sp_col2:
        if star2:
            star2_injured = not st.toggle(
                f"Available — {star2['star_player']}", value=True, key="sp2")
            st.markdown(star_card_html(team2, star2, star2_injured), unsafe_allow_html=True)
            if star2_injured:
                st.warning(f"⚠️ Playing without **{star2['star_player']}**")
        else:
            star2_injured = False
            st.caption(f"No star player data for {team2}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Recent Form ───────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Recent International Form")
    st.caption("Last 5 international matches before 2026 World Cup")
    rf_col1, rf_col2 = st.columns(2)
    with rf_col1:
        outcomes1 = get_last5_results(team1)
        form1 = get_recent_form(team1)
        arrow1, acolor1 = form_trend(team1)
        st.markdown(f"**{flag_team(team1)}**")
        st.markdown(form_dots_html(outcomes1), unsafe_allow_html=True)
        st.markdown(
            f'<span style="font-size:1.05rem;font-weight:600;color:#1a5c2e;">Form: {form1:.0%}</span>'
            f'&nbsp;&nbsp;<span style="font-size:1.3rem;font-weight:700;color:{acolor1};">{arrow1}</span>',
            unsafe_allow_html=True)
    with rf_col2:
        outcomes2 = get_last5_results(team2)
        form2 = get_recent_form(team2)
        arrow2, acolor2 = form_trend(team2)
        st.markdown(f"**{flag_team(team2)}**")
        st.markdown(form_dots_html(outcomes2), unsafe_allow_html=True)
        st.markdown(
            f'<span style="font-size:1.05rem;font-weight:600;color:#1a5c2e;">Form: {form2:.0%}</span>'
            f'&nbsp;&nbsp;<span style="font-size:1.3rem;font-weight:700;color:{acolor2};">{arrow2}</span>',
            unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Head to Head History ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚔️ Head to Head History")
    st.caption("World Cup meetings only (1930 – 2022)")
    try:
        show_h2h_section(team1, team2, load_match_history())
    except Exception as e:
        st.error(f"Could not load H2H history: {e}")

    # ── Prediction output ─────────────────────────────────────────────────────
    if predict_btn:
        if team1 == team2:
            st.warning("Please select two different teams.")
        else:
            try:
                stage_num = STAGE_NUM_MAP[stage]
                with st.spinner("Predicting…"):
                    X_row   = build_feature_row(team1, team2, rank1, rank2,
                                                stage_num, stats, h2h_dict)
                    p1_base, p2_base = predict_prob(model, X_row)
                p1, p2 = apply_star_penalty(p1_base, p2_base, star1_injured, star2_injured)

                st.markdown("---")
                f1, f2 = get_flag(team1), get_flag(team2)
                st.markdown(
                    f'<div style="text-align:center;font-size:1.6rem;font-weight:700;'
                    f'color:#1a5c2e;margin-bottom:0.5rem;">'
                    f'{f1} {team1} &nbsp;vs&nbsp; {f2} {team2}</div>',
                    unsafe_allow_html=True)
                st.markdown("### Prediction Results")

                if star1_injured or star2_injured:
                    bc1, bc2, bc3 = st.columns(3)
                    with bc1:
                        st.markdown(
                            f'<div style="background:#fff9e6;border-radius:10px;padding:10px 14px;'
                            f'text-align:center;border:1px solid #f0c040;">'
                            f'<div style="font-size:0.8rem;color:#888;">Base probability</div>'
                            f'<div style="font-size:1.1rem;font-weight:700;">{f1} {team1}: {p1_base}%</div>'
                            f'<div style="font-size:1.1rem;font-weight:700;">{f2} {team2}: {p2_base}%</div>'
                            f'</div>', unsafe_allow_html=True)
                    with bc2:
                        st.markdown(
                            '<div style="text-align:center;font-size:2rem;padding-top:10px;">→</div>',
                            unsafe_allow_html=True)
                    with bc3:
                        st.markdown(
                            f'<div style="background:#fdecea;border-radius:10px;padding:10px 14px;'
                            f'text-align:center;border:1px solid #e57373;">'
                            f'<div style="font-size:0.8rem;color:#888;">After injury adjustment</div>'
                            f'<div style="font-size:1.1rem;font-weight:700;">{f1} {team1}: {p1}%</div>'
                            f'<div style="font-size:1.1rem;font-weight:700;">{f2} {team2}: {p2}%</div>'
                            f'</div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                g1, mid, g2 = st.columns([3, 1, 3])
                with g1:
                    st.plotly_chart(make_gauge(p1, f"{f1} {team1}"),
                                    use_container_width=True, key="g1")
                with mid:
                    st.markdown(
                        '<div style="text-align:center;font-size:2.5rem;color:#1a5c2e;'
                        'font-weight:700;padding-top:80px">VS</div>', unsafe_allow_html=True)
                with g2:
                    st.plotly_chart(make_gauge(p2, f"{f2} {team2}"),
                                    use_container_width=True, key="g2")

                winner_label = team1 if p1 > p2 else (team2 if p2 > p1 else "Draw")
                winner_flag  = get_flag(winner_label) if winner_label != "Draw" else ""
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(
                        f'<div class="metric-box">'
                        f'<div class="label">{f1} {team1} Win Probability</div>'
                        f'<div class="value">{p1}%</div></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(
                        f'<div class="metric-box">'
                        f'<div class="label">{f2} {team2} Win Probability</div>'
                        f'<div class="value">{p2}%</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                if winner_label != "Draw":
                    st.success(f"🏆 Predicted Winner: **{winner_flag} {winner_label}**")
                else:
                    st.info("Match is predicted to be too close to call.")

                # ── Historical record vs model alignment ──────────────────────
                _h2h_pred = get_h2h_stats(team1, team2, load_match_history())
                if _h2h_pred and _h2h_pred["total"] > 0:
                    hist_win_rate = round(_h2h_pred["team1_wins"] / _h2h_pred["total"] * 100, 1)
                    st.markdown(
                        f'<div style="background:#f8f9fa;border-radius:8px;padding:10px 16px;'
                        f'margin-top:8px;font-size:0.88rem;">'
                        f'📜 Historical record: <b>{get_flag(team1)} {team1}</b> has won '
                        f'<b>{hist_win_rate}%</b> of their {_h2h_pred["total"]} World Cup '
                        f'meeting{"s" if _h2h_pred["total"]>1 else ""} against '
                        f'<b>{get_flag(team2)} {team2}</b>.</div>',
                        unsafe_allow_html=True)
                    model_favours_t1 = p1 > p2
                    hist_favours_t1  = hist_win_rate > 50
                    if model_favours_t1 == hist_favours_t1:
                        st.success("✅ Model prediction **aligns** with the historical record.")
                    else:
                        st.warning(
                            "⚠️ Model prediction **differs** from history — "
                            "potential upset scenario.")

                with st.expander("🔍 Feature values used for this prediction"):
                    st.dataframe(
                        X_row.rename(columns={
                            "rank_diff": "Rank Diff", "team1_win_rate": f"{team1} Win Rate",
                            "team2_win_rate": f"{team2} Win Rate", "head_to_head": "H2H Win Rate",
                            "stage_num": "Stage", "team1_avg_goals": f"{team1} Avg Goals",
                            "team2_avg_goals": f"{team2} Avg Goals", "team1_is_host": "Host",
                            "coach_exp_diff": "Coach Exp Diff",
                            "team1_conf_strength": f"{team1} Conf", "team2_conf_strength": f"{team2} Conf",
                            "conf_strength_diff": "Conf Diff",
                            "team1_recent_form": f"{team1} Form", "team2_recent_form": f"{team2} Form",
                            "form_diff": "Form Diff",
                        }).style.format("{:.3f}"), use_container_width=True)

                st.markdown("---")
                show_shap_section()
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
    else:
        st.info("👆 Configure the match above and click **Predict Win Probability** to get started.")
        show_shap_section()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — 2026 Fixture Predictions
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "2026 Fixture Predictions":
    st.markdown("# 🗓️ FIFA World Cup 2026 — Fixture Predictions")
    st.markdown("Predicted win probabilities for all confirmed group-stage fixtures.")
    st.markdown("---")

    fixtures = load_fixtures()
    sp_df = load_star_players()

    # ── Build base predictions (cached) ───────────────────────────────────────
    @st.cache_data
    def predict_all_fixtures():
        rows = []
        for _, row in fixtures.iterrows():
            t1, t2 = row["HomeTeamName"], row["AwayTeamName"]
            r1 = int(row["home_rank"]) if pd.notna(row["home_rank"]) else 35
            r2 = int(row["away_rank"]) if pd.notna(row["away_rank"]) else 35
            X = build_feature_row(t1, t2, r1, r2, 0, stats, h2h_dict)
            p1, p2 = predict_prob(model, X)
            try:
                date_str = pd.to_datetime(row["kickoff_at"], utc=True).strftime("%b %d, %Y")
            except Exception:
                date_str = str(row["kickoff_at"])[:10]
            rows.append({
                "Match": int(row["match_number"]), "Date": date_str,
                "Group": f"Group {row['group']}",
                "Team 1": flag_team(t1), "Team 1 Win%": p1,
                "Team 2": flag_team(t2), "Team 2 Win%": p2,
                "Predicted Winner": flag_team(t1) if p1 > p2 else flag_team(t2),
                "_t1": t1, "_t2": t2,   # raw names for injury lookup
                "home_rank": r1, "away_rank": r2,
            })
        return pd.DataFrame(rows)

    pred_df = predict_all_fixtures()

    # ── Player Availability Settings ──────────────────────────────────────────
    with st.expander("⭐ Player Availability Settings", expanded=False):
        st.caption("Toggle off any star player marked as injured or suspended. "
                   "Probabilities will be recalculated with a 15% reduction for that team.")
        # Use raw team names (stored in _t1/_t2) for the availability toggles
        all_fixture_teams = sorted(set(
            pred_df["_t1"].tolist() + pred_df["_t2"].tolist()
        ))

        injured_teams = set()   # raw team names
        cols_per_row = 4
        for i in range(0, len(all_fixture_teams), cols_per_row):
            row_teams = all_fixture_teams[i:i + cols_per_row]
            avail_cols = st.columns(cols_per_row)
            for j, team in enumerate(row_teams):
                sp_row = sp_df[sp_df["team"] == team]
                if not sp_row.empty:
                    player = sp_row.iloc[0]["star_player"]
                    pos    = sp_row.iloc[0]["position"]
                    init   = sp_row.iloc[0]["availability"] == "available"
                    with avail_cols[j]:
                        is_avail = st.toggle(
                            f"{flag_team(team)}", value=init,
                            key=f"fix_sp_{team}",
                            help=f"{player} · {pos}",
                        )
                        st.caption(f"⭐ {player}")
                        if not is_avail:
                            injured_teams.add(team)

    # ── Apply injury adjustments (compare raw names via _t1/_t2) ─────────────
    display_df = pred_df.copy()
    for idx, row in display_df.iterrows():
        t1_inj = row["_t1"] in injured_teams
        t2_inj = row["_t2"] in injured_teams
        if t1_inj or t2_inj:
            p1a, p2a = apply_star_penalty(row["Team 1 Win%"], row["Team 2 Win%"],
                                           t1_inj, t2_inj)
            display_df.at[idx, "Team 1 Win%"] = p1a
            display_df.at[idx, "Team 2 Win%"] = p2a
            display_df.at[idx, "Predicted Winner"] = (
                flag_team(row["_t1"]) if p1a > p2a else flag_team(row["_t2"])
            )

    if injured_teams:
        inj_display = ", ".join(f"{flag_team(t)} {t}" for t in sorted(injured_teams))
        st.info(f"⚠️ Injury adjustments applied for: **{inj_display}**")

    # ── Filters ───────────────────────────────────────────────────────────────
    fc1, fc2 = st.columns([2, 2])
    with fc1:
        groups = ["All Groups"] + sorted(display_df["Group"].unique())
        sel_group = st.selectbox("Filter by Group", groups)
    with fc2:
        sel_stage = st.selectbox("Filter by Stage", ["All Stages", "Group Stage"])

    filtered = display_df.copy()
    if sel_group != "All Groups":
        filtered = filtered[filtered["Group"] == sel_group]

    st.markdown(f"Showing **{len(filtered)}** matches")
    st.markdown("---")

    # ── Styled table ──────────────────────────────────────────────────────────
    display_cols = ["Match", "Date", "Group", "Team 1", "Team 1 Win%",
                    "Team 2", "Team 2 Win%", "Predicted Winner"]
    show_df = filtered[display_cols].reset_index(drop=True)

    def style_winner(row):
        styles = [""] * len(row)
        winner = row["Predicted Winner"]
        win_style = "background-color:#d4edda;font-weight:600;color:#155724;"
        if winner == row["Team 1"]:
            styles[display_cols.index("Team 1")]          = win_style
            styles[display_cols.index("Predicted Winner")] = win_style
        else:
            styles[display_cols.index("Team 2")]          = win_style
            styles[display_cols.index("Predicted Winner")] = win_style
        return styles

    styled = (
        show_df.style
        .apply(style_winner, axis=1)
        .format({"Team 1 Win%": "{:.1f}%", "Team 2 Win%": "{:.1f}%"})
        .set_properties(**{"font-size": "0.9rem"})
        .set_table_styles([
            {"selector": "th", "props": [("background-color","#1a5c2e"),("color","white"),
                                         ("font-size","0.9rem"),("font-weight","600"),
                                         ("text-align","center")]},
            {"selector": "td", "props": [("text-align","center"),("padding","6px 10px")]},
        ])
        .bar(subset=["Team 1 Win%"], color="#b8dfc8", vmin=0, vmax=100)
        .bar(subset=["Team 2 Win%"], color="#f4c2c2", vmin=0, vmax=100)
    )
    st.dataframe(styled, use_container_width=True, height=600)

    # ── H2H lookup for any fixture ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚔️ Head to Head Lookup")
    st.caption("Select a fixture to view full World Cup head-to-head history.")
    fixture_labels = [
        f"{row['Team 1']} vs {row['Team 2']}"
        for _, row in filtered.iterrows()
    ]
    if fixture_labels:
        sel_fix = st.selectbox("Select a fixture", fixture_labels, label_visibility="collapsed")
        sel_idx = fixture_labels.index(sel_fix)
        sel_row = filtered.iloc[sel_idx]
        t1_raw, t2_raw = sel_row["_t1"], sel_row["_t2"]
        with st.expander(f"View H2H: {sel_fix}", expanded=True):
            try:
                show_h2h_section(t1_raw, t2_raw, load_match_history(), compact=True)
            except Exception as e:
                st.error(f"Could not load H2H history: {e}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Summary")
    s1, s2, s3 = st.columns(3)
    avg_conf  = filtered["Team 1 Win%"].apply(lambda x: abs(x - 50)).mean()
    home_wins = (filtered["Predicted Winner"] == filtered["Team 1"]).sum()
    away_wins = (filtered["Predicted Winner"] == filtered["Team 2"]).sum()
    with s1:
        st.markdown(
            f'<div class="metric-box"><div class="label">Avg Model Confidence</div>'
            f'<div class="value">{50 + avg_conf:.1f}%</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(
            f'<div class="metric-box"><div class="label">Predicted Home Wins</div>'
            f'<div class="value">{home_wins}</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(
            f'<div class="metric-box"><div class="label">Predicted Away Wins</div>'
            f'<div class="value">{away_wins}</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Tournament Bracket Simulator
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Tournament Bracket":
    st.markdown("# 🏆 Tournament Bracket Simulator")
    st.markdown("🤖 Simulate the full FIFA World Cup 2026 using the ML model")
    st.markdown("---")

    # Collect injured teams from star_players CSV (used when star_impact is on)
    _sp_df  = load_star_players()
    inj_set = set(_sp_df[_sp_df["availability"] != "available"]["team"].tolist()) if star_impact else set()

    # Pre-load base probabilities (cached)
    with st.spinner("Loading probability matrix…"):
        base_probs = precompute_base_probs()

    # ── Run Simulation button ─────────────────────────────────────────────────
    run_btn = st.button("▶ Run Full Simulation", type="primary", use_container_width=False)

    if run_btn or "bracket_result" in st.session_state:
        if run_btn:
            with st.spinner("Simulating tournament…"):
                result = simulate_tournament(upset_factor, star_impact, inj_set,
                                             int(sim_seed), base_probs)
            st.session_state["bracket_result"] = result
        else:
            result = st.session_state["bracket_result"]

        gs     = result["group_standings"]
        rounds = result["rounds"]
        champ  = result["champion"]
        path   = result["path"]

        # ── SECTION 2 — Group Stage ───────────────────────────────────────────
        st.markdown("## Group Stage Results")
        group_keys = list(GROUPS.keys())
        for row_start in range(0, len(group_keys), 3):
            cols = st.columns(3)
            for ci, gk in enumerate(group_keys[row_start:row_start+3]):
                standing = gs[gk]
                with cols[ci]:
                    st.markdown(f"**Group {gk}**")
                    rows_html = ""
                    bg_map = {1:"#d4edda", 2:"#d4edda", 3:"#fff3cd", 4:"#f8d7da"}
                    status = {1:"✅", 2:"✅", 3:"⚠️", 4:"❌"}
                    for entry in standing:
                        bg = bg_map[entry["rank"]]
                        rows_html += (
                            f'<tr style="background:{bg};">'
                            f'<td style="padding:4px 6px;">{status[entry["rank"]]}</td>'
                            f'<td style="padding:4px 6px;font-weight:600;">'
                            f'{flag_team(entry["team"])}</td>'
                            f'<td style="padding:4px 8px;text-align:center;">'
                            f'<b>{entry["pts"]}</b></td>'
                            f'</tr>'
                        )
                    st.markdown(
                        '<table style="width:100%;border-collapse:collapse;'
                        'font-size:0.82rem;margin-bottom:12px;">'
                        '<thead><tr style="background:#1a5c2e;color:white;">'
                        '<th style="padding:4px 6px;"></th>'
                        '<th style="padding:4px 6px;">Team</th>'
                        '<th style="padding:4px 8px;">Pts</th>'
                        f'</tr></thead><tbody>{rows_html}</tbody></table>',
                        unsafe_allow_html=True)

        # ── SECTION 3 — Bracket / Knockout Rounds ─────────────────────────────
        st.markdown("---")
        st.markdown("## Knockout Bracket")

        round_labels = {
            "R32": "Round of 32", "R16": "Round of 16",
            "QF":  "Quarter-Finals", "SF": "Semi-Finals", "Final": "Final"
        }
        for rname, rlabel in round_labels.items():
            matches = rounds[rname]
            with st.expander(f"**{rlabel}** — {len(matches)} match{'es' if len(matches)>1 else ''}", expanded=(rname=="Final")):
                ncols = min(4, len(matches))
                for chunk_start in range(0, len(matches), ncols):
                    chunk = matches[chunk_start:chunk_start+ncols]
                    mcols = st.columns(len(chunk))
                    for ci, m in enumerate(chunk):
                        with mcols[ci]:
                            st.markdown(match_card(m), unsafe_allow_html=True)

        # ── SECTION 4 — Champion Prediction Card ─────────────────────────────
        st.markdown("---")
        champ_flag = get_flag(champ)
        toughest   = min(path, key=lambda x: x["win_prob"])
        easiest    = max(path, key=lambda x: x["win_prob"])
        overall_p  = round(np.prod([x["win_prob"]/100 for x in path]) * 100, 1)

        st.markdown(
            f'<div style="background:linear-gradient(135deg,#1a5c2e,#2e7d32);'
            f'color:white;border-radius:16px;padding:2rem;text-align:center;margin-bottom:1.5rem;">'
            f'<div style="font-size:1rem;opacity:0.8;margin-bottom:0.4rem;">🏆 Predicted Champion</div>'
            f'<div style="font-size:3rem;font-weight:800;margin-bottom:0.5rem;">'
            f'{champ_flag} {champ}</div>'
            f'<div style="font-size:0.9rem;opacity:0.85;">'
            f'Combined path probability: {overall_p}%</div>'
            f'</div>', unsafe_allow_html=True)

        p1c, p2c, p3c = st.columns(3)
        with p1c:
            st.markdown("**🛤️ Path to Glory**")
            for step in path:
                st.markdown(
                    f'`{step["round"]}` {flag_team(step["opponent"])} — '
                    f'**{step["win_prob"]}%**')
        with p2c:
            st.markdown("**😰 Toughest Match**")
            st.markdown(
                f'{flag_team(toughest["opponent"])}\n\n'
                f'Round: `{toughest["round"]}`  \nWin prob: **{toughest["win_prob"]}%**')
        with p3c:
            st.markdown("**😎 Easiest Match**")
            st.markdown(
                f'{flag_team(easiest["opponent"])}\n\n'
                f'Round: `{easiest["round"]}`  \nWin prob: **{easiest["win_prob"]}%**')

        # ── SECTION 5 — Tournament Statistics ─────────────────────────────────
        st.markdown("---")
        st.markdown("## Tournament Statistics")

        all_matches = [m for r in rounds.values() for m in r]
        closest   = min(all_matches, key=lambda m: abs(m["p_t1"] - 50))
        biggest   = max(all_matches, key=lambda m: m["p_winner"])

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f'<div class="metric-box"><div class="label">🏆 Champion</div>'
                f'<div class="value" style="font-size:1.1rem;">{champ_flag} {champ}</div></div>',
                unsafe_allow_html=True)
        with s2:
            st.markdown(
                f'<div class="metric-box"><div class="label">🎲 Most Competitive Match</div>'
                f'<div class="value" style="font-size:0.9rem;">'
                f'{flag_team(closest["t1"])} vs {flag_team(closest["t2"])}'
                f'<br><span style="font-size:0.75rem;">{closest["p_t1"]}% / {round(100-closest["p_t1"],1)}%</span>'
                f'</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(
                f'<div class="metric-box"><div class="label">💪 Most Dominant Win</div>'
                f'<div class="value" style="font-size:0.9rem;">'
                f'{flag_team(biggest["winner"])}<br>'
                f'<span style="font-size:0.75rem;">{biggest["p_winner"]}% confidence</span>'
                f'</div></div>', unsafe_allow_html=True)
        with s4:
            total_matches = sum(len(v) for v in rounds.values())
            st.markdown(
                f'<div class="metric-box"><div class="label">📊 Total Matches</div>'
                f'<div class="value">{total_matches}</div></div>',
                unsafe_allow_html=True)

        # ── SECTION 6 — 100 Simulations ───────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🔄 Multi-Simulation Analysis")
        n_sims_btn = st.button("🔄 Run 100 Simulations", use_container_width=False)

        if n_sims_btn or "sim100_result" in st.session_state:
            if n_sims_btn:
                with st.spinner("Running 100 simulations… (may take ~20s)"):
                    champ_counts, final_pairs = run_n_simulations(
                        100, upset_factor, star_impact, inj_set, base_probs)
                st.session_state["sim100_result"] = (champ_counts, final_pairs)
            else:
                champ_counts, final_pairs = st.session_state["sim100_result"]

            top10 = sorted(champ_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_names = [flag_team(t) for t, _ in top10]
            top_pcts  = [round(c, 1) for _, c in top10]
            bar_colors= [get_conf_color(t) for t, _ in top10]

            fig_sim = go.Figure(go.Bar(
                x=top_pcts, y=top_names, orientation="h",
                marker_color=bar_colors,
                text=[f"{p}%" for p in top_pcts], textposition="outside",
            ))
            fig_sim.update_layout(
                title="Championship Probability — 100 Simulations",
                xaxis_title="Times Won Champion (%)",
                yaxis={"autorange": "reversed"},
                height=420, margin=dict(l=200, r=60, t=50, b=40),
                plot_bgcolor="#f4f9f4", paper_bgcolor="#f4f9f4",
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            # Most likely final
            top_final = sorted(final_pairs.items(), key=lambda x: x[1], reverse=True)[:3]
            st.markdown("**Most Likely Final Matchups:**")
            for (a, b), cnt in top_final:
                st.markdown(f"- {flag_team(a)} vs {flag_team(b)} — **{cnt}×** out of 100")

        # ── Download Button ────────────────────────────────────────────────────
        st.markdown("---")
        csv_rows = []
        for rname, matches in rounds.items():
            for m in matches:
                csv_rows.append({
                    "Round": rname, "Team1": m["t1"], "Team2": m["t2"],
                    "Winner": m["winner"],
                    "Team1_WinProb": m["p_t1"],
                    "Winner_Confidence": m["p_winner"],
                })
        csv_df = pd.DataFrame(csv_rows)
        st.download_button(
            "📥 Download Bracket as CSV",
            data=csv_df.to_csv(index=False),
            file_name="wc2026_bracket_simulation.csv",
            mime="text/csv",
        )

    else:
        st.info("👆 Click **▶ Run Full Simulation** to simulate the complete FIFA World Cup 2026 bracket.")
        st.markdown("**Groups:**")
        gcols = st.columns(4)
        for ci, (gk, teams) in enumerate(GROUPS.items()):
            with gcols[ci % 4]:
                st.markdown(f"**Group {gk}**")
                for t in teams:
                    st.markdown(f"  {flag_team(t)}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Star Players 2026
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Star Players 2026":
    st.markdown("# ⭐ Star Players — FIFA World Cup 2026")
    st.markdown("Star player availability status for all 2026 World Cup teams.")
    st.markdown("---")

    sp_df = load_star_players().copy()
    sp_df["confederation"] = sp_df["team"].apply(get_conf)

    # ── Summary stats ─────────────────────────────────────────────────────────
    total_avail   = (sp_df["availability"] == "available").sum()
    total_injured = (sp_df["availability"] != "available").sum()
    conf_injured  = (sp_df[sp_df["availability"] != "available"]["confederation"]
                     .value_counts().idxmax()
                     if total_injured > 0 else "None")

    sm1, sm2, sm3 = st.columns(3)
    with sm1:
        st.markdown(
            f'<div class="metric-box"><div class="label">Available Players</div>'
            f'<div class="value" style="color:#a8e6b4;">{total_avail}</div></div>',
            unsafe_allow_html=True)
    with sm2:
        st.markdown(
            f'<div class="metric-box"><div class="label">Injured / Suspended</div>'
            f'<div class="value" style="color:#f4a0a0;">{total_injured}</div></div>',
            unsafe_allow_html=True)
    with sm3:
        st.markdown(
            f'<div class="metric-box"><div class="label">Most Affected Confederation</div>'
            f'<div class="value" style="font-size:1.3rem;">{conf_injured}</div></div>',
            unsafe_allow_html=True)

    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns([3, 2, 2])
    with f1:
        search = st.text_input("🔍 Search by team or player name", "")
    with f2:
        conf_options = ["All Confederations"] + sorted(sp_df["confederation"].unique().tolist())
        sel_conf = st.selectbox("Filter by Confederation", conf_options)
    with f3:
        avail_options = ["All", "Available", "Injured / Suspended"]
        sel_avail = st.selectbox("Filter by Availability", avail_options)

    filtered_sp = sp_df.copy()
    if search:
        mask = (
            filtered_sp["team"].str.contains(search, case=False, na=False) |
            filtered_sp["star_player"].str.contains(search, case=False, na=False)
        )
        filtered_sp = filtered_sp[mask]
    if sel_conf != "All Confederations":
        filtered_sp = filtered_sp[filtered_sp["confederation"] == sel_conf]
    if sel_avail == "Available":
        filtered_sp = filtered_sp[filtered_sp["availability"] == "available"]
    elif sel_avail == "Injured / Suspended":
        filtered_sp = filtered_sp[filtered_sp["availability"] != "available"]

    st.markdown(f"Showing **{len(filtered_sp)}** teams")
    st.markdown("---")

    # ── Styled table ──────────────────────────────────────────────────────────
    def render_star_table(df):
        rows_html = ""
        for _, r in df.iterrows():
            conf  = r["confederation"]
            badge_bg, badge_lbl = CONF_BADGE.get(conf, ("#546E7A", conf))
            avail_bg  = "#2e7d32" if r["availability"] == "available" else "#c62828"
            avail_txt = "Available" if r["availability"] == "available" else "Injured / Suspended"
            pos_colors = {"Forward":"#e65100","Midfielder":"#1565c0",
                          "Defender":"#2e7d32","Goalkeeper":"#6a1b9a"}
            pos_bg = pos_colors.get(r["position"], "#555")
            rows_html += (
                f"<tr>"
                f'<td style="font-weight:600;">{get_flag(r["team"])} {r["team"]}'
                f'&nbsp;<span style="background:{badge_bg};color:#fff;padding:1px 7px;'
                f'border-radius:8px;font-size:0.7rem;">{badge_lbl}</span></td>'
                f'<td style="font-weight:600;">{r["star_player"]}</td>'
                f'<td><span style="background:{pos_bg};color:#fff;padding:1px 7px;'
                f'border-radius:8px;font-size:0.72rem;">{r["position"]}</span></td>'
                f'<td>{r["club"]}</td>'
                f'<td><span style="background:{avail_bg};color:#fff;padding:2px 10px;'
                f'border-radius:8px;font-size:0.78rem;font-weight:600;">{avail_txt}</span></td>'
                f"</tr>"
            )
        return (
            '<table style="width:100%;border-collapse:collapse;font-size:0.88rem;">'
            '<thead><tr style="background:#1a5c2e;color:white;">'
            '<th style="padding:8px 12px;text-align:left;">Team</th>'
            '<th style="padding:8px 12px;text-align:left;">Star Player</th>'
            '<th style="padding:8px 12px;text-align:left;">Position</th>'
            '<th style="padding:8px 12px;text-align:left;">Club</th>'
            '<th style="padding:8px 12px;text-align:left;">Status</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody></table>'
        )

    st.markdown(render_star_table(filtered_sp), unsafe_allow_html=True)

    # ── Impact Analysis ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Availability Impact Analysis")
    st.caption("How much each team's win probability drops when their star player is absent (15% penalty model).")

    fixtures = load_fixtures()

    @st.cache_data
    def compute_impact():
        rows = []
        for _, fix in fixtures.iterrows():
            t1, t2 = fix["HomeTeamName"], fix["AwayTeamName"]
            r1 = int(fix["home_rank"]) if pd.notna(fix["home_rank"]) else 35
            r2 = int(fix["away_rank"]) if pd.notna(fix["away_rank"]) else 35
            X  = build_feature_row(t1, t2, r1, r2, 0, stats, h2h_dict)
            p1, p2 = predict_prob(model, X)
            # Impact on team1 star
            p1_inj, _ = apply_star_penalty(p1, p2, True, False)
            rows.append({"team": t1, "base_prob": p1, "inj_prob": p1_inj,
                         "drop": round(p1 - p1_inj, 1), "opponent": t2,
                         "match_label": f"{t1} vs {t2}"})
            # Impact on team2 star
            _, p2_inj = apply_star_penalty(p1, p2, False, True)
            rows.append({"team": t2, "base_prob": p2, "inj_prob": p2_inj,
                         "drop": round(p2 - p2_inj, 1), "opponent": t1,
                         "match_label": f"{t2} vs {t1}"})
        return pd.DataFrame(rows)

    impact_df = compute_impact()

    # Average drop per team
    team_impact = (
        impact_df.groupby("team")["drop"]
        .mean().reset_index()
        .rename(columns={"drop": "avg_prob_drop"})
        .sort_values("avg_prob_drop", ascending=False)
        .head(20)
    )
    team_impact["confederation"] = team_impact["team"].apply(get_conf)
    team_impact["color"] = team_impact["confederation"].apply(
        lambda c: CONF_BADGE.get(c, ("#546E7A",""))[0])

    fig_bar = go.Figure(go.Bar(
        x=team_impact["avg_prob_drop"],
        y=team_impact["team"],
        orientation="h",
        marker_color=team_impact["color"],
        text=team_impact["avg_prob_drop"].apply(lambda v: f"{v:.1f}pp"),
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Average Win Probability Drop Without Star Player (Top 20 Teams)",
        xaxis_title="Avg Probability Drop (percentage points)",
        yaxis={"autorange": "reversed"},
        height=520,
        margin=dict(l=160, r=80, t=50, b=40),
        plot_bgcolor="#f4f9f4", paper_bgcolor="#f4f9f4",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Top 5 most impacted matches
    st.markdown("#### Top 5 Matches Most Affected by Star Player Absence")
    top5_matches = (
        impact_df.sort_values("drop", ascending=False)
        .drop_duplicates("match_label")
        .head(5)[["match_label", "team", "base_prob", "inj_prob", "drop"]]
        .rename(columns={
            "match_label": "Match", "team": "Missing Player's Team",
            "base_prob": "Base Win%", "inj_prob": "With Injury Win%",
            "drop": "Prob Drop (pp)"
        })
        .reset_index(drop=True)
    )
    st.dataframe(
        top5_matches.style
        .format({"Base Win%": "{:.1f}%", "With Injury Win%": "{:.1f}%",
                 "Prob Drop (pp)": "{:.1f}"})
        .background_gradient(subset=["Prob Drop (pp)"], cmap="Reds"),
        use_container_width=True,
    )

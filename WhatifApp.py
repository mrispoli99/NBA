import streamlit as st
import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Pro Analytics Simulator", layout="wide", page_icon="🏀")

# --- 2. DATA ENGINE ---
@st.cache_data
def load_data():
    try:
        t = pd.read_excel('teams.xlsx')
        p = pd.read_excel('players.xlsx')
        if 'Games_Played' not in p.columns:
            st.error("⚠️ Please run updatefiles.py to fetch Games Played data.")
            return None, None
        return t, p
    except: return None, None

TEAMS_DF, PLAYERS_DF = load_data()
if TEAMS_DF is None: st.stop()

def get_team_stats(team): return TEAMS_DF[TEAMS_DF['Team'] == team].iloc[0]

# --- INTELLIGENCE LOGIC ---
def get_position_category(pos):
    if pos in ['PG', 'SG', 'G']: return 'Def_vs_Guard'
    if pos in ['SF', 'PF', 'F']: return 'Def_vs_Wing'
    return 'Def_vs_Big'

def simulate_player_performance(row, opp_stats, pace_factor, is_home):
    if not row['Available'] or row['Exp Minutes'] <= 0: return None
    
    # BASE CALCULATIONS
    avg_mins = row['Avg_Mins'] if row['Avg_Mins'] > 5 else 36.0
    usage_rate = row['Exp Minutes'] / avg_mins
    
    matchup = opp_stats.get(get_position_category(row['Position']), 1.0)
    
    variance_range = 0.10 
    if row['Games_Played'] < 20: variance_range = 0.25 
    elif row['Games_Started'] > 30: variance_range = 0.05 
    
    noise = np.random.uniform(1.0 - variance_range, 1.0 + variance_range)
    
    # --- NEW: USAGE HIERARCHY (The Alpha Dog Boost) ---
    # Superstars demand the ball. We boost their base volume.
    usage_boost = 1.0
    if row['Avg_Pts'] >= 25: usage_boost = 1.10   # Superstars
    elif row['Avg_Pts'] >= 20: usage_boost = 1.05 # All-Stars
    
    # Role players shoot better at home
    home_boost = 1.05 if (is_home and row['Avg_Pts'] < 15) else 1.0
    
    # Calculate final multiplier (removed the 0.95 universal penalty to help scores)
    mult = usage_rate * pace_factor * matchup * noise * home_boost * usage_boost
    
    return {
        "Player": row['Player'],
        "PTS": float(row['Avg_Pts'] * mult), # Keep as float for math later
        "REB": float(row['Avg_Reb'] * mult),
        "AST": float(row['Avg_Ast'] * mult),
        "STL": float(row['Avg_Stl'] * mult),
        "BLK": float(row['Avg_Blk'] * mult),
        "TOV": float(row['Avg_Tov'] * mult),
        "Exp Minutes": row['Exp Minutes'],
        "Avg_Pts": row['Avg_Pts']
    }

def normalize_stats(df):
    if df.empty: return df
    total_mins = df['Exp Minutes'].sum()
    
    cols = ['PTS','REB','AST','STL','BLK','TOV']
    
    # CASE 1: OVER-ALLOCATION (>242 mins)
    # If the user leaves too many players active, we must scale down.
    if total_mins > 242:
        scale = 240.0 / total_mins
        st.toast(f"⚠️ Minutes High ({int(total_mins)}). Bench scaled down to protect stars.")
        
        # --- NEW: STAR PROTECTION ---
        # Instead of scaling everyone equally, we protect the high-scorers
        stars = df['Avg_Pts'] >= 22
        role_players = df['Avg_Pts'] < 22
        
        # Stars only take 30% of the penalty. Role players take 150% of the penalty.
        star_scale = 1.0 - ((1.0 - scale) * 0.3)
        role_scale = 1.0 - ((1.0 - scale) * 1.5)
        
        # Prevent role players from going negative
        role_scale = max(role_scale, 0.1)
        
        for c in cols:
            df.loc[stars, c] *= star_scale
            df.loc[role_players, c] *= role_scale

    # CASE 2: UNDER-ALLOCATION (<238 mins)
    elif total_mins < 238:
        scale = 240.0 / total_mins
        is_thin = scale > 1.25
        fatigue = 0.92 if is_thin else 1.0
        
        if is_thin: st.toast("⚠️ Roster Thin. Star Usage + Fatigue Active!")
        else: st.toast("⚠️ Minutes Low. Auto-Filled.")
        
        for c in cols: df[c] = df[c] * scale * fatigue
        
        # Hero Ball for thin rosters
        if is_thin:
            stars = df['Avg_Pts'] >= 20
            df.loc[stars, 'PTS'] *= 1.15 # Massive boost
            df.loc[stars, 'TOV'] *= 1.15
            df.loc[~stars, 'PTS'] *= 0.90
            
    # Round everything cleanly
    for c in cols: df[c] = df[c].round().astype(int)
    return df

# --- UI SECTION ---
st.title("NBA Game Simulator for BW SPM Class (Spring 2026)")

# TEAM SELECTORS
c1, _, c2 = st.columns([1, 0.1, 1])
with c1:
    h_team = st.selectbox("Home", TEAMS_DF['Team'].unique(), index=0)
    h_wins = st.slider(f"{h_team} Wins (L10)", 0, 10, 5)
    h_b2b = st.checkbox("Home B2B?", key="h_b2b")
with c2:
    a_team = st.selectbox("Away", TEAMS_DF['Team'].unique(), index=1)
    a_wins = st.slider(f"{a_team} Wins (L10)", 0, 10, 5)
    a_b2b = st.checkbox("Away B2B?", key="a_b2b")

st.divider()

# --- INTELLIGENT ROSTER EDITOR ---
def create_smart_editor(team, key):
    df = PLAYERS_DF[PLAYERS_DF['Team'] == team].copy()
    
    df['Available'] = df['Games_Played'] > 15
    df['Exp Minutes'] = df['Avg_Mins']
    
    conditions = [
        (df['Games_Played'] < 10),
        (df['Games_Started'] > df['Games_Played'] * 0.8)
    ]
    choices = ['⚠️ Low Sample', '⭐ Starter']
    df['Role'] = np.select(conditions, choices, default='Bench')

    st.markdown(f"**{team} Rotation**")
    edited = st.data_editor(
        df,
        column_order=["Available", "Exp Minutes", "Player", "Role", "Games_Played"],
        column_config={
            "Available": st.column_config.CheckboxColumn("Active?", width="small"),
            "Exp Minutes": st.column_config.NumberColumn("Mins", max_value=48, width="small"),
            "Role": st.column_config.TextColumn("Role", disabled=True, width="small"),
            "Games_Played": st.column_config.ProgressColumn("Games", min_value=0, max_value=82, format="%d", width="small"),
        },
        disabled=["Player", "Role", "Games_Played"],
        hide_index=True,
        key=key,
        use_container_width=True
    )
    return edited

c1, c2 = st.columns(2)
with c1: h_roster = create_smart_editor(h_team, "h_edit")
with c2: a_roster = create_smart_editor(a_team, "a_edit")

# --- SIMULATION ---
if st.button("🔮 SIMULATE", type="primary", use_container_width=True):
    h_stats = get_team_stats(h_team)
    a_stats = get_team_stats(a_team)
    pace = (h_stats['Pace'] + a_stats['Pace']) / 200.0
    
    gap = h_wins - a_wins
    if abs(gap) >= 5:
        fav_roster = h_roster if gap > 0 else a_roster
        fav_roster.loc[fav_roster['Avg_Mins'] > 28, 'Exp Minutes'] *= 0.85
        st.toast("⚠️ Blowout Risk: Starters resting.")

    h_box = [simulate_player_performance(r, a_stats, pace, True) for _, r in h_roster.iterrows()]
    a_box = [simulate_player_performance(r, h_stats, pace, False) for _, r in a_roster.iterrows()]
    
    h_df = normalize_stats(pd.DataFrame([x for x in h_box if x]))
    a_df = normalize_stats(pd.DataFrame([x for x in a_box if x]))
    
    h_score = h_df['PTS'].sum() + 3 + max(0, h_df['REB'].sum() - a_df['REB'].sum())*0.5
    a_score = a_df['PTS'].sum() + max(0, a_df['REB'].sum() - h_df['REB'].sum())*0.5
    
    st.balloons()
    st.markdown(f"<h1 style='text-align:center'>{h_team} {int(h_score)} - {int(a_score)} {a_team}</h1>", unsafe_allow_html=True)
    
    t1, t2 = st.tabs([h_team, a_team])
    with t1: st.dataframe(h_df.style.background_gradient(cmap="Greens", subset=['PTS']), use_container_width=True)
    with t2: st.dataframe(a_df.style.background_gradient(cmap="Reds", subset=['PTS']), use_container_width=True)
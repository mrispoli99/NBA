import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Pro Analytics Simulator", layout="wide", page_icon="🏀")

@st.cache_data
def load_data():
    try:
        t = pd.read_excel('teams.xlsx')
        p = pd.read_excel('players.xlsx')
        # Ensure we have new columns
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
    
    # INTELLIGENT NOISE:
    # If they play < 20 games, they are "High Variance" (Unreliable)
    # If they start > 30 games, they are "Consistent"
    variance_range = 0.10 # Standard +/- 10%
    if row['Games_Played'] < 20: variance_range = 0.25 # +/- 25% (Wildcard)
    elif row['Games_Started'] > 30: variance_range = 0.05 # +/- 5% (Steady)
    
    noise = np.random.uniform(1.0 - variance_range, 1.0 + variance_range)
    
    # Home Boost for Role Players
    home_boost = 1.05 if (is_home and row['Avg_Pts'] < 15) else 1.0
    
    mult = usage_rate * pace_factor * matchup * noise * home_boost * 0.95
    
    return {
        "Player": row['Player'],
        "PTS": int(row['Avg_Pts'] * mult),
        "REB": int(row['Avg_Reb'] * mult),
        "AST": int(row['Avg_Ast'] * mult),
        "STL": int(row['Avg_Stl'] * mult),
        "BLK": int(row['Avg_Blk'] * mult),
        "TOV": int(row['Avg_Tov'] * mult),
        "Exp Minutes": row['Exp Minutes'],
        "Avg_Pts": row['Avg_Pts']
    }

def normalize_stats(df):
    if df.empty: return df
    total = df['Exp Minutes'].sum()
    
    if total > 242:
        scale = 240.0 / total
        st.toast(f"⚠️ Minutes High ({int(total)}). Scaled Down.")
        for c in ['PTS','REB','AST','STL','BLK','TOV']: df[c] = (df[c] * scale).astype(int)
        
    elif total < 238:
        scale = 240.0 / total
        is_thin = scale > 1.25
        fatigue = 0.92 if is_thin else 1.0
        
        if is_thin: st.toast("⚠️ Roster Thin. Hero Ball Active!")
        else: st.toast("⚠️ Minutes Low. Auto-Filled.")
        
        for c in ['PTS','REB','AST','STL','BLK','TOV']: df[c] = (df[c] * scale * fatigue).astype(int)
        
        if is_thin:
            stars = df['Avg_Pts'] >= 20
            df.loc[stars, 'PTS'] = (df.loc[stars, 'PTS'] * 1.1).astype(int)
            df.loc[stars, 'TOV'] = (df.loc[stars, 'TOV'] * 1.1).astype(int)
            
    return df

# --- UI SECTION ---
st.title("🏀 Game Strategy Simulator 2.0")

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
    
    # SMART DEFAULTING:
    # Only mark Active if they have played > 15 games
    # (Fixes the "12 minutes but only played twice" bug)
    df['Available'] = df['Games_Played'] > 15
    df['Exp Minutes'] = df['Avg_Mins']
    
    # Create a "Status" text for context
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
    
    # Blowout Logic
    gap = h_wins - a_wins
    if abs(gap) >= 5:
        fav_roster = h_roster if gap > 0 else a_roster
        fav_roster.loc[fav_roster['Avg_Mins'] > 28, 'Exp Minutes'] *= 0.85
        st.toast("⚠️ Blowout Risk: Starters resting.")

    # Run Sim
    h_box = [simulate_player_performance(r, a_stats, pace, True) for _, r in h_roster.iterrows()]
    a_box = [simulate_player_performance(r, h_stats, pace, False) for _, r in a_roster.iterrows()]
    
    h_df = normalize_stats(pd.DataFrame([x for x in h_box if x]))
    a_df = normalize_stats(pd.DataFrame([x for x in a_box if x]))
    
    # Rebounding & Scoring
    h_score = h_df['PTS'].sum() + 3 + max(0, h_df['REB'].sum() - a_df['REB'].sum())*0.5
    a_score = a_df['PTS'].sum() + max(0, a_df['REB'].sum() - h_df['REB'].sum())*0.5
    
    # Display
    st.balloons()
    st.markdown(f"<h1 style='text-align:center'>{h_team} {int(h_score)} - {int(a_score)} {a_team}</h1>", unsafe_allow_html=True)
    
    t1, t2 = st.tabs([h_team, a_team])
    with t1: st.dataframe(h_df.style.background_gradient(cmap="Greens", subset=['PTS']), use_container_width=True)
    with t2: st.dataframe(a_df.style.background_gradient(cmap="Reds", subset=['PTS']), use_container_width=True)
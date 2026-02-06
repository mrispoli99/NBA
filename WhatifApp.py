import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors # Helper for color gradients if needed

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NBA Pro Analytics Simulator for BW SPM244", layout="wide", page_icon="🏀")

# --- 2. DATA ENGINE ---
@st.cache_data
def load_data():
    try:
        teams_df = pd.read_excel('teams.xlsx')
        players_df = pd.read_excel('players.xlsx')
        
        req_cols = ['Avg_Pts', 'Avg_Reb', 'Avg_Ast', 'Avg_Stl', 'Avg_Blk', 'Avg_Tov']
        if not all(col in players_df.columns for col in req_cols):
            st.error("⚠️ Data Error: players.xlsx is missing stat columns. Please run updatefiles.py.")
            return None, None
            
        return teams_df, players_df
    except FileNotFoundError:
        st.error("❌ Critical Error: Excel files not found. Please make sure teams.xlsx and players.xlsx are in this folder.")
        return None, None

TEAMS_DF, PLAYERS_DF = load_data()

if TEAMS_DF is None: st.stop()

def get_team_stats(team_name):
    return TEAMS_DF[TEAMS_DF['Team'] == team_name].iloc[0]

# --- 3. SIMULATION LOGIC ---
def get_position_category(pos):
    if pos in ['PG', 'SG', 'G']: return 'Def_vs_Guard'
    if pos in ['SF', 'PF', 'F']: return 'Def_vs_Wing'
    if pos in ['C']: return 'Def_vs_Big'
    return 'Def_vs_Wing' 

def simulate_player_performance(row, opp_stats, pace_factor):
    if not row['Available'] or row['Exp Minutes'] <= 0:
        return None
        
    avg_mins = row['Avg_Mins'] if row['Avg_Mins'] > 5 else 36.0 
    usage_rate = row['Exp Minutes'] / avg_mins
    
    def_col = get_position_category(row['Position'])
    matchup_multiplier = opp_stats.get(def_col, 1.0)
    
    # Noise Factors
    noise = np.random.uniform(0.90, 1.10)
    def_noise = np.random.uniform(0.80, 1.20)
    system_efficiency = 0.95 # General friction
    
    base_mult = usage_rate * pace_factor * matchup_multiplier * noise * system_efficiency
    
    stats = {
        "Player": row['Player'],
        "PTS": row['Avg_Pts'] * base_mult, # Keep as float for now
        "REB": row['Avg_Reb'] * base_mult,
        "AST": row['Avg_Ast'] * base_mult,
        "STL": row['Avg_Stl'] * base_mult * def_noise,
        "BLK": row['Avg_Blk'] * base_mult * def_noise,
        "TOV": row['Avg_Tov'] * base_mult,
        "Exp Minutes": row['Exp Minutes'],
        "Avg_Pts": row['Avg_Pts'] # Needed for Hero Ball detection
    }
    return stats

def normalize_stats(df):
    """
    Smart Normalization:
    1. Scale to regulation minutes (240).
    2. Apply Fatigue Penalty if roster is thin.
    3. Apply 'Hero Ball' Bonus to stars if roster is thin.
    """
    if df.empty: return df
    
    total_mins = df['Exp Minutes'].sum()
    
    # CASE 1: OVER-ALLOCATION (>242 mins)
    if total_mins > 242:
        scale_factor = 240.0 / total_mins
        st.toast(f"⚠️ Minutes High ({int(total_mins)}). Scaled down.")
        
        cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        for c in cols:
            df[c] = (df[c] * scale_factor)

    # CASE 2: UNDER-ALLOCATION (<238 mins) -> The "OKC Fix"
    elif total_mins < 238:
        scale_factor = 240.0 / total_mins
        
        # Logic: If scale > 1.25 (roster is very thin), apply penalties/bonuses
        is_thin_roster = scale_factor > 1.25
        fatigue_penalty = 0.92 if is_thin_roster else 1.0
        
        if is_thin_roster:
            st.toast(f"⚠️ Roster Thin ({int(total_mins)} mins). Scaling Up + Hero Ball Active!")
        else:
            st.toast(f"⚠️ Minutes Low ({int(total_mins)}). Auto-filled minutes.")

        # Apply Base Scaling + Fatigue
        cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        for c in cols:
            df[c] = df[c] * scale_factor * fatigue_penalty
            
        # HERO BALL LOGIC 🦸‍♂️
        if is_thin_roster:
            # Find Stars: Players who average > 20 PPG
            # We give them a usage boost (taking shots from the G-Leaguers)
            star_mask = df['Avg_Pts'] >= 20
            
            # Boost their Points by 10% (Volume scoring)
            # Increase their Turnovers by 10% (Forced passes)
            df.loc[star_mask, 'PTS'] *= 1.10
            df.loc[star_mask, 'TOV'] *= 1.10
            
            # Reduce others slightly to balance (optional, but realistic)
            df.loc[~star_mask, 'PTS'] *= 0.95

    # Final Rounding to Integers
    cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    for c in cols:
        df[c] = df[c].round().astype(int)
        
    return df

# --- NEW: BLOWOUT LOGIC ---
def apply_blowout_logic(roster_df, wins_gap):
    adjusted_roster = roster_df.copy()
    if wins_gap >= 5:
        starters_mask = adjusted_roster['Avg_Mins'] > 28
        bench_mask = (adjusted_roster['Avg_Mins'] <= 28) & (adjusted_roster['Avg_Mins'] > 0)
        adjusted_roster.loc[starters_mask, 'Exp Minutes'] *= 0.85
        adjusted_roster.loc[bench_mask, 'Exp Minutes'] *= 1.20
        return adjusted_roster, True
    return adjusted_roster, False

# --- 4. UI: SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Commissioner Tools")
    if st.button("🔄 Reload Database"):
        st.cache_data.clear()
        st.rerun()
    st.info("Update your Excel files and click here to refresh the app.")

# --- 5. UI: MAIN SETUP ---
st.title("🏀 NBA Game Strategy Simulator (For BW SPM244)")

st.markdown("### 1. Matchup Configuration")
st.info("📝 **Instructions:** Select your teams below. Be sure to update the **Last 10 Wins** slider (to reflect current momentum) and check the **Back-to-Back** box if the team played yesterday.")

c1, c2, c3 = st.columns([1, 0.1, 1])

with c1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox("Select Team", TEAMS_DF['Team'].unique(), index=0, key="h_sel")
    h_wins = st.slider(f"{home_team} Wins (Last 10)", 0, 10, 5)
    h_b2b = st.checkbox("Back-to-Back Game?", key="h_b2b")

with c3:
    st.subheader("✈️ Away Team")
    default_idx = 1 if len(TEAMS_DF) > 1 else 0
    away_team = st.selectbox("Select Team", TEAMS_DF['Team'].unique(), index=default_idx, key="a_sel")
    a_wins = st.slider(f"{away_team} Wins (Last 10)", 0, 10, 5)
    a_b2b = st.checkbox("Back-to-Back Game?", key="a_b2b")

if home_team == away_team:
    st.warning("⚠️ Please select two different teams.")
    st.stop()

st.divider()

# --- 6. UI: ROSTER MANAGEMENT ---
st.markdown("### 2. Roster Management")
st.info("📝 **Instructions:** Check the box on the left to mark a player **Active**. Click the **Mins** column to adjust their playing time for this specific matchup.")

col_h, col_a = st.columns(2)

def create_roster_editor(team, key):
    df = PLAYERS_DF[PLAYERS_DF['Team'] == team].copy()
    df['Available'] = True
    df['Exp Minutes'] = df['Avg_Mins']
    
    st.markdown(f"**{team} Rotation**")
    
    edited_df = st.data_editor(
        df,
        column_order=["Available", "Exp Minutes", "Player", "Position"],
        column_config={
            "Available": st.column_config.CheckboxColumn("Active?", default=True, width="small"),
            "Exp Minutes": st.column_config.NumberColumn("Mins", min_value=0, max_value=48, width="small"),
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Position": st.column_config.TextColumn("Pos", width="small"),
        },
        disabled=["Player", "Team", "Position", "Avg_Pts"],
        hide_index=True,
        key=key,
        use_container_width=True
    )
    return edited_df

with col_h:
    home_roster = create_roster_editor(home_team, "h_edit")

with col_a:
    away_roster = create_roster_editor(away_team, "a_edit")

# --- 7. EXECUTION ENGINE ---
st.divider()
run_sim = st.button("🔮 SIMULATE GAME", type="primary", use_container_width=True)

if run_sim:
    h_stats = get_team_stats(home_team)
    a_stats = get_team_stats(away_team)
    
    # 1. Blowout Logic
    h_roster_final, h_blowout = apply_blowout_logic(home_roster, h_wins - a_wins)
    a_roster_final, a_blowout = apply_blowout_logic(away_roster, a_wins - h_wins)
    
    if h_blowout: st.warning(f"⚠️ **Mismatch:** {home_team} favored. Starters resting in 4th.")
    if a_blowout: st.warning(f"⚠️ **Mismatch:** {away_team} favored. Starters resting in 4th.")

    # 2. Pace
    game_pace = (h_stats['Pace'] + a_stats['Pace']) / 2
    pace_factor = game_pace / 100.0
    
    # 3. Run Sim
    h_box_rows = [simulate_player_performance(row, a_stats, pace_factor) for i, row in h_roster_final.iterrows()]
    a_box_rows = [simulate_player_performance(row, h_stats, pace_factor) for i, row in a_roster_final.iterrows()]
    
    h_box_rows = [x for x in h_box_rows if x is not None]
    a_box_rows = [x for x in a_box_rows if x is not None]
    
    h_df = pd.DataFrame(h_box_rows)
    a_df = pd.DataFrame(a_box_rows)
    
    # 4. Normalize (Includes "Hero Ball")
    h_df = normalize_stats(h_df)
    a_df = normalize_stats(a_df)
    
    # 5. Totals
    h_points = h_df['PTS'].sum()
    a_points = a_df['PTS'].sum()
    
    h_intangibles = h_stats['Home_Advantage'] + (h_wins - 5) - (3 if h_b2b else 0)
    a_intangibles = (a_wins - 5) - (3 if a_b2b else 0)
    
    final_h_score = int(h_points + h_intangibles)
    final_a_score = int(a_points + a_intangibles)
    
    spread = final_h_score - final_a_score
    
    st.balloons()
    
    st.markdown(f"""
    <div style="text-align: center; border: 2px solid #444; padding: 20px; border-radius: 10px; background-color: #262730;">
        <h2>FINAL SCORE PROJECTION</h2>
        <h1 style="font-size: 60px; margin: 0;">
            <span style="color: #4CAF50;">{home_team} {final_h_score}</span> 
            <span style="color: #888;">-</span> 
            <span style="color: #FF5252;">{away_team} {final_a_score}</span>
        </h1>
        <p><i>Projected Pace: {game_pace:.1f} possessions</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📊 Projected Box Scores")
    
    tab1, tab2 = st.tabs([f"{home_team} Stats", f"{away_team} Stats"])
    
    with tab1:
        st.dataframe(
            h_df.set_index('Player').style.background_gradient(cmap="Greens", subset=['PTS']), 
            use_container_width=True
        )
        st.caption(f"Team Turnovers: {h_df['TOV'].sum()} | Team Steals: {h_df['STL'].sum()}")
        
    with tab2:
        st.dataframe(
            a_df.set_index('Player').style.background_gradient(cmap="Reds", subset=['PTS']), 
            use_container_width=True
        )
        st.caption(f"Team Turnovers: {a_df['TOV'].sum()} | Team Steals: {a_df['STL'].sum()}")

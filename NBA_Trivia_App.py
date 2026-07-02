import streamlit as st
import pandas as pd
import os
import time

st.set_page_config(page_title="NBA Category Timeline Trivia", layout="centered")

# 1. CLEAN LOADING ENGINE
@st.cache_data
def load_game_data():
    csv_path = "nba_trivia_data.csv"
    if not os.path.exists(csv_path): 
        return None, [], [], pd.DataFrame()
    
    df = pd.read_csv(csv_path).fillna("N/A")
    df = df.drop_duplicates(subset=['Year'], keep='first')
    
    all_players = set()
    for col in ['MVP', 'DPOY', 'Finals MVP', 'Scoring Leader', 'Assists Leader', 'Rebound Leader']:
        all_players.update(df[col].astype(str).unique())
        
    # FIX: Explicit list of historical/modern NBA team abbreviations to ensure thoroughness
    global_teams = [
        "76ers", "Blazers", "Bombers", "Bucks", "Bulls", "Bullets", "Capitols", "Cavs", 
        "Celtics", "Grizzlies", "Hawks", "Heat", "Jazz", "Kings", "Knicks", "Lakers", 
        "Magic", "Mavs", "Nets", "Nuggets", "Pacers", "Packers", "Pelicans", "Pistons", 
        "Raptors", "Rockets", "Suns", "Spurs", "Sonics", "Stags", "Thunder", "Timberwolves", 
        "Warriors", "Wizards"
    ]
    
    clean_players = sorted([p for p in all_players if p.lower() not in ['nan', 'n/a', '']])
    
    return df, clean_players, global_teams

df, global_players, global_teams = load_game_data()

if df is None or df.empty:
    st.error("⚠️ 'nba_trivia_data.csv' not found or empty! Please run your scraper script first.")
    st.stop()

# 2. DEFINE GAME CFG MAPS
game_modes = {
    "NBA Scoring leader": {"col": "Scoring Leader", "type": "player", "start_year": 1948},
    "NBA finals winner": {"col": "Champion", "type": "team", "start_year": 1948},
    "NBA finals runner up": {"col": "Runner-Up", "type": "team", "start_year": 1948},
    "NBA MVP": {"col": "MVP", "type": "player", "start_year": 1956},
    "NBA defensive player of the year": {"col": "DPOY", "type": "player", "start_year": 1983},
    "NBA Finals MVP": {"col": "Finals MVP", "type": "player", "start_year": 1969},
    "NBA Assists leader": {"col": "Assists Leader", "type": "player", "start_year": 1948, "limited_options": True},
    "NBA Rebound leader": {"col": "Rebound Leader", "type": "player", "start_year": 1951, "limited_options": True}
}

st.sidebar.title("🎮 Select Game Mode")
selected_game = st.sidebar.radio("Pick a stat line to solve:", list(game_modes.keys()))

game_cfg = game_modes[selected_game]
target_col = game_cfg["col"]
filtered_df = df[df['Year'] >= game_cfg["start_year"]].sort_values(by="Year", ascending=False)

# Reset state when shifting game modes
if "active_game" not in st.session_state or st.session_state.active_game != selected_game:
    st.session_state.active_game = selected_game
    st.session_state.attempts = 0
    st.session_state.game_over = False
    st.session_state.feedback = {}
    st.session_state.start_time = None  
    st.session_state.time_expired = False

st.title(f"🏆 {selected_game} Timeline")
timer_placeholder = st.empty()

# --- 3. FIX: ACCURATE TIMER START CHECK ---
# Scan inputs to see if a real selection change happened (ignoring default/unfilled states)
current_inputs = [v for k, v in st.session_state.items() if k.startswith(f"input_{selected_game}_") and v not in [None, "", "Choose..."]]

if current_inputs and st.session_state.start_time is None and not st.session_state.game_over:
    st.session_state.start_time = time.time()

# 4. ISOLATED TIMER FRAGMENT
@st.fragment(run_every=1.0)
def render_live_timer():
    if st.session_state.start_time is not None and not st.session_state.game_over:
        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, 300 - int(elapsed)) 
        
        if remaining <= 0:
            st.session_state.time_expired = True
            st.session_state.game_over = True
            st.session_state.attempts = 3
            st.parent_rerun() 
            
        mins, secs = divmod(remaining, 60)
        timer_placeholder.error(f"⏱️ **TIME REMAINING: {mins}:{secs:02d}**")
    elif st.session_state.game_over and st.session_state.time_expired:
        timer_placeholder.error("⏰ TIME EXPIRED! You took longer than 5 minutes and lost all attempts.")
    elif st.session_state.start_time is None:
        timer_placeholder.info("⏱️ **The 5-minute timer countdown will start the exact second you make your first guess below!**")

render_live_timer()

# 5. MAIN FORM WORKSPACE
with st.form("timeline_form"):
    st.write(f"### Attempt {st.session_state.attempts}/3")
    
    user_guesses = {}
    
    for idx, row in filtered_df.iterrows():
        year = int(row['Year'])
        
        if game_cfg.get("limited_options"):
            past_winners = df[(df['Year'] <= year) & (df['Year'] >= game_cfg["start_year"])].sort_values(by="Year", ascending=False)
            recent_options = past_winners[target_col].astype(str).unique()[:5]
            dropdown_options = sorted(list(recent_options))
        else:
            dropdown_options = global_teams if game_cfg["type"] == "team" else global_players
            
        guess = st.selectbox(
            f"Year {year}",
            options=dropdown_options,
            index=None,
            placeholder="Choose...",
            key=f"input_{selected_game}_{year}_{idx}",
            disabled=st.session_state.game_over
        )
        user_guesses[year] = guess or ""
        
    submit_btn = st.form_submit_button("Submit Entire List", disabled=st.session_state.game_over)

# 6. EVALUATION LOGIC
if submit_btn:
    st.session_state.attempts += 1
    
    results = {}
    all_correct = True
    
    for idx, row in filtered_df.iterrows():
        year = int(row['Year'])
        actual = str(row[target_col]).lower().strip()
        user_val = str(user_guesses.get(year, "")).lower().strip()
        
        is_correct = (user_val == actual)
        results[year] = is_correct
        if not is_correct:
            all_correct = False
            
    st.session_state.feedback = results
    
    if all_correct:
        st.success(f"🎉 Incredible! You cleared the entire timeline perfectly on attempt {st.session_state.attempts}!")
        st.session_state.game_over = True
        st.rerun()
    elif st.session_state.attempts >= 3:
        st.error("❌ Out of attempts! The correct answers have been revealed below.")
        st.session_state.game_over = True
        st.rerun()
    else:
        st.warning(f"Some answers are incorrect. You have {3 - st.session_state.attempts} attempt(s) remaining!")
        st.rerun()

# 7. RESULTS OVERVIEW (FIXED TO DELAY REVEALING ANSWERS)
if st.session_state.feedback:
    st.write("---")
    st.write("### Review Your List Status:")
    
    for idx, row in filtered_df.iterrows():
        year = int(row['Year'])
        actual = str(row[target_col])
        is_ok = st.session_state.feedback.get(year, False)
        
        if is_ok:
            st.markdown(f"✅ **{year}:** Correct")
        else:
            # FIX: Only reveal the actual answer string value if game_over is set to True
            reveal = f" *(Answer: {actual})*".replace("N/A", "No Award This Year") if st.session_state.game_over else ""
            st.markdown(f"❌ **{year}:** Incorrect{reveal}")
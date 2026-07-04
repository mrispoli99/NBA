import streamlit as st
import pandas as pd
import os
import time
import random

st.set_page_config(page_title="NBA Complete Trivia Arena", layout="centered")

# 1. LOAD DATA ENGINE
@st.cache_data
def load_game_data():
    csv_path = "nba_trivia_data.csv"
    if not os.path.exists(csv_path): return None, [], [], pd.DataFrame()
    df = pd.read_csv(csv_path).fillna("N/A").drop_duplicates(subset=['Year'], keep='first')
    
    all_players = set()
    for col in ['MVP', 'DPOY', 'Finals MVP', 'ROY', 'Scoring Leader', 'Assists Leader', 'Rebound Leader']:
        all_players.update(df[col].astype(str).unique())
        
    global_teams = [
        "76ers", "Blazers", "Bombers", "Bucks", "Bulls", "Bullets", "Capitols", "Cavs", 
        "Celtics", "Grizzlies", "Hawks", "Heat", "Jazz", "Kings", "Knicks", "Lakers", 
        "Magic", "Mavs", "Nets", "Nuggets", "Pacers", "Packers", "Pelicans", "Pistons", 
        "Raptors", "Rockets", "Suns", "Spurs", "Sonics", "Stags", "Thunder", "Timberwolves", 
        "Warriors", "Wizards"
    ]
    return df, sorted([p for p in all_players if p.lower() not in ['nan', 'n/a', '']]), global_teams

df, global_players, global_teams = load_game_data()
if df is None or df.empty:
    st.error("⚠️ 'nba_trivia_data.csv' not found. Run 'build_nba_data.py' first.")
    st.stop()

# 2. DEFINE MAPS FOR THE GAME MODES
game_modes = {
    "⚡ LIGHTNING RAPID FIRE": {"col": "SPECIAL", "type": "mixed", "start_year": 1948},
    "NBA Rookie of the year": {"col": "ROY", "type": "player", "start_year": 1953},
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
selected_game = st.sidebar.radio("Pick a trivia challenge:", list(game_modes.keys()))
game_cfg = game_modes[selected_game]

# Reset configurations upon changing menus
if "active_game" not in st.session_state or st.session_state.active_game != selected_game:
    st.session_state.active_game = selected_game
    st.session_state.attempts = 0
    st.session_state.game_over = False
    st.session_state.feedback = {}
    st.session_state.start_time = None  
    st.session_state.time_expired = False
    # Specific variables for lightning mode
    st.session_state.lt_started = False
    st.session_state.lt_chosen_metrics = []
    st.session_state.lt_max_questions = 30  # Default fallback
    st.session_state.lt_correct = 0
    st.session_state.lt_total = 0
    st.session_state.lt_current_q = None
    st.session_state.lt_last_feedback = ""

st.title(f"🏆 {selected_game}")
timer_placeholder = st.empty()

# 3. GLOBAL TIMER FRAGMENT (7 MINUTES)
@st.fragment(run_every=1.0)
def render_live_timer():
    if st.session_state.start_time is not None and not st.session_state.game_over:
        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, 420 - int(elapsed))
        if remaining <= 0:
            st.session_state.time_expired = True
            st.session_state.game_over = True
            st.parent_rerun()
        mins, secs = divmod(remaining, 60)
        timer_placeholder.error(f"⏱️ **TIME REMAINING: {mins}:{secs:02d}**")
    elif st.session_state.game_over and st.session_state.time_expired:
        timer_placeholder.error("⏰ TIME EXPIRED! The clock hit 7 minutes. See your final summary scorecard below.")

render_live_timer()

# --- HELPER FUNCTION FOR ACCURATE GRADING MESSAGES ---
def render_grading_message(correct, total):
    pct = int((correct / total) * 100) if total > 0 else 0
    st.write("---")
    st.write("### 🏁 Final Game Summary")
    st.info(f"**Final Score:** {correct} / {total} Correct ({pct}%)")
    if pct == 100:
        st.balloons(); st.success("👑 **LEGENDARY STATUS!** Flawless game! Your basketball IQ is historic.")
    elif pct >= 85:
        st.success("🔥 **CHAMPION PERFORMANCE!** Outstanding job! You're elite.")
    elif pct >= 50:
        st.warning("💪 **SOLID EFFORT!** Nice work! You passed, but study up for elite status.")
    else:
        st.error("🧱 **AIRBALL!** Tough game. Hit the tape, practice, and try again!")

# ==========================================
# BRANCH A: LIGHTNING RAPID FIRE GAME LOOP
# ==========================================
if selected_game == "⚡ LIGHTNING RAPID FIRE":
    # 1. SETUP PHASE
    if not st.session_state.lt_started:
        st.write("### ⚙️ Configure Your Blitz Round")
        st.markdown("Choose your custom pools and length limit below. The 7-minute timer will not start until you press the launch button.")
        
        # FIX: "NBA Rookie of the year" is now allowed and listed in the options pool
        available_metrics = [
            k for k in game_modes.keys() if k != "⚡ LIGHTNING RAPID FIRE"
        ]
        
        chosen_metrics = st.multiselect(
            "Metrics to include:",
            options=available_metrics,
            default=available_metrics,
            placeholder="Select at least one metric..."
        )
        
        # NEW: Round limit selector configuration
        chosen_limit = st.selectbox(
            "Number of questions for this round:",
            options=[25, 30, 40, 50],
            index=1  # Defaults to 30
        )
        
        start_blitz = st.button("🚀 Start Blitz Game", disabled=len(chosen_metrics) == 0)
        
        if start_blitz:
            st.session_state.lt_chosen_metrics = chosen_metrics
            st.session_state.lt_max_questions = chosen_limit
            st.session_state.lt_started = True
            st.session_state.start_time = time.time()  
            st.rerun()

    # 2. ACTION PHASE
    else:
        # Check if we hit the user-selected question cap
        if st.session_state.lt_total >= st.session_state.lt_max_questions:
            st.session_state.game_over = True

        if st.session_state.game_over:
            if st.session_state.lt_total >= st.session_state.lt_max_questions and not st.session_state.time_expired:
                st.success(f"🎯 **Completed all {st.session_state.lt_max_questions} questions!** Check your final stats:")
            render_grading_message(st.session_state.lt_correct, st.session_state.lt_total)
        else:
            # Generate questions dynamically using only chosen keys
            if st.session_state.lt_current_q is None:
                q_mode = random.choice(st.session_state.lt_chosen_metrics)
                cfg = game_modes[q_mode]
                possible_years = df[df['Year'] >= cfg["start_year"]]['Year'].tolist()
                q_year = random.choice(possible_years)
                
                row = df[df['Year'] == q_year].iloc[0]
                st.session_state.lt_current_q = {
                    "year": q_year, "mode_name": q_mode, "col": cfg["col"], "type": cfg["type"],
                    "correct_ans": str(row[cfg["col"]]), "limited_options": cfg.get("limited_options", False)
                }
            
            q = st.session_state.lt_current_q
            st.subheader(f"Question {st.session_state.lt_total + 1} of {st.session_state.lt_max_questions}")
            st.markdown(f"### Guess the **{q['mode_name']}** for the year **{q['year']}**")
            
            if q.get("limited_options"):
                past_winners = df[(df['Year'] <= q['year']) & (df['Year'] >= game_modes[q['mode_name']]["start_year"])].sort_values(by="Year", ascending=False)
                dropdown_options = sorted(list(past_winners[q['col']].astype(str).unique()[:5]))
            else:
                dropdown_options = global_teams if q["type"] == "team" else global_players

            with st.form("lightning_form", clear_on_submit=True):
                user_guess = st.selectbox("Your Answer:", options=dropdown_options, index=None, placeholder="Type to filter...")
                submit_ans = st.form_submit_button("Submit Answer")
                
            if submit_ans:
                st.session_state.lt_total += 1
                actual = q["correct_ans"].lower().strip()
                guessed = str(user_guess or "").lower().strip()
                
                if guessed == actual:
                    st.session_state.lt_correct += 1
                    st.session_state.lt_last_feedback = f"✅ **Correct!** The answer was *{q['correct_ans']}*."
                else:
                    st.session_state.lt_last_feedback = f"❌ **Incorrect.** You guessed *{user_guess}*. The answer was *{q['correct_ans']}*."
                    
                st.session_state.lt_current_q = None  
                st.rerun()
                
            if st.session_state.lt_last_feedback:
                st.markdown(st.session_state.lt_last_feedback)

# ==========================================
# BRANCH B: REGULAR TIMELINE LIST GAME MODES
# ==========================================
else:
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    filtered_df = df[df['Year'] >= game_cfg["start_year"]].sort_values(by="Year", ascending=False)
    
    with st.form("timeline_form"):
        st.write(f"### Attempt {st.session_state.attempts}/3")
        user_guesses = {}
        
        for i, (idx, row) in enumerate(filtered_df.iterrows()):
            year = int(row['Year'])
            if game_cfg.get("limited_options"):
                past_winners = df[(df['Year'] <= year) & (df['Year'] >= game_cfg["start_year"])].sort_values(by="Year", ascending=False)
                dropdown_options = sorted(list(past_winners[target_col].astype(str).unique()[:5]))
            else:
                dropdown_options = global_teams if game_cfg["type"] == "team" else global_players
                
            guess = st.selectbox(
                f"Year {year}", options=dropdown_options, index=None,
                placeholder="Choose...", key=f"{selected_game}_{year}_{i}", disabled=st.session_state.game_over
            )
            user_guesses[year] = guess or ""
            
        submit_btn = st.form_submit_button("Submit Entire List", disabled=st.session_state.game_over)

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
            if not is_correct: all_correct = False
                
        st.session_state.feedback = results
        if all_correct or st.session_state.attempts >= 3:
            st.session_state.game_over = True
        st.rerun()

    # Score Overview Render
    if st.session_state.feedback:
        total_items = len(st.session_state.feedback)
        correct_count = sum(1 for v in st.session_state.feedback.values() if v is True)
        
        if st.session_state.game_over:
            render_grading_message(correct_count, total_items)
        else:
            st.write(f"### 📝 Current List Status (Attempts Remaining: {3 - st.session_state.attempts}):")
            
        for idx, row in filtered_df.iterrows():
            year = int(row['Year'])
            actual = str(row[target_col])
            is_ok = st.session_state.feedback.get(year, False)
            if is_ok:
                st.markdown(f"✅ **{year}:** Correct")
            else:
                reveal = f" *(Answer: {actual})*".replace("N/A", "No Award This Year") if st.session_state.game_over else ""
                st.markdown(f"❌ **{year}:** Incorrect{reveal}")
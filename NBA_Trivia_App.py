import streamlit as st
import pandas as pd
import os
import time
import random
from thefuzz import process

st.set_page_config(page_title="NBA Complete Trivia Arena", layout="centered")

# ==========================================
# 1. CORE DATA LOADING ENGINE
# ==========================================
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

@st.cache_data
def load_hof_list():
    if os.path.exists("nba_hof_players.csv"):
        return pd.read_csv("nba_hof_players.csv")['HOF_Player'].tolist()
    return []

@st.cache_data
def load_player_metadata():
    if os.path.exists("nba_player_metadata.csv"):
        pmdf = pd.read_csv("nba_player_metadata.csv").fillna("N/A")
        return pmdf, pmdf['Player'].tolist()
    return pd.DataFrame(), []

# Fixes for team names that fell through the scraper's mapping before it was
# extended (see TEAM_MAP in build_nba_data.py) -- applied here too so an
# already-generated nba_decade_rosters.csv gets corrected without re-scraping.
TEAM_NAME_FIXES = {
    'Buffalo Braves': 'Braves', 'Capital Bullets': 'Bullets',
    'Charlotte Bobcats': 'Bobcats', 'Charlotte Hornets': 'Hornets',
    'Chicago Packers': 'Packers', 'Chicago Zephyrs': 'Zephyrs',
    'Cincinnati Royals': 'Kings', 'Kansas City Kings': 'Kings', 'Kansas City-Omaha Kings': 'Kings',
    'Los Angeles Clippers': 'Clippers', 'San Diego Clippers': 'Clippers',
    'Memphis Grizzlies': 'Grizzlies', 'Vancouver Grizzlies': 'Grizzlies',
    'Minnesota Timberwolves': 'Timberwolves',
    'New Orleans Hornets': 'Hornets', 'New Orleans/Oklahoma City Hornets': 'Hornets',
    'New Orleans Jazz': 'Jazz', 'New Orleans Pelicans': 'Pelicans',
    'New York Nets': 'Nets', 'San Diego Rockets': 'Rockets', 'San Francisco Warriors': 'Warriors',
}

@st.cache_data
def load_decade_rosters():
    if os.path.exists("nba_decade_rosters.csv"):
        rdf = pd.read_csv("nba_decade_rosters.csv").fillna("")
        rdf['Team'] = rdf['Team'].replace(TEAM_NAME_FIXES)
        return rdf
    return pd.DataFrame()

df, global_players, global_teams = load_game_data()
hof_master_list = load_hof_list()
player_meta_df, player_meta_master_list = load_player_metadata()
decade_rosters_df = load_decade_rosters()

if df is None or df.empty:
    st.error("⚠️ 'nba_trivia_data.csv' not found. Please run your scraper script first.")
    st.stop()

# ==========================================
# 2. DEFINE MAPS FOR THE GAME MODES
# ==========================================
game_modes = {
    "🏠 HOME SCREEN": {"col": "NONE", "type": "meta", "start_year": 0},
    "⚡ LIGHTNING RAPID FIRE": {"col": "SPECIAL", "type": "mixed", "start_year": 1948},
    "🏛️ HOF NAMING SPRINT": {"col": "HOF", "type": "text_sprint", "start_year": 0},
    "🔍 PLAYER ID LIGHTNING": {"col": "PLAYER_META", "type": "text_sprint", "start_year": 0},
    "🕵️ MYSTERY ROSTER": {"col": "ROSTER", "type": "text_sprint", "start_year": 0},
    "📋 ROSTER RECALL LIGHTNING": {"col": "ROSTER_RECALL", "type": "text_sprint", "start_year": 0},
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

# --- TWO-WAY NAVIGATION CONTROLLER SYSTEM ---
if "nav_state" not in st.session_state:
    st.session_state.nav_state = "🏠 HOME SCREEN"

# Look up the correct index position matching our master state tracking string
modes_list = list(game_modes.keys())
current_idx = modes_list.index(st.session_state.nav_state)

st.sidebar.title("🎮 Main Navigation")
selected_game = st.sidebar.radio(
    "Go to:", 
    options=modes_list, 
    index=current_idx
)

# Sync state tracker seamlessly if navigation triggers via sidebar manually
if selected_game != st.session_state.nav_state:
    st.session_state.nav_state = selected_game
    st.rerun()

active_selection = st.session_state.nav_state
game_cfg = game_modes[active_selection]

# --- SESSION STATE INITIALIZATION & STATE REBOOTS ---
if "active_game" not in st.session_state or st.session_state.active_game != active_selection:
    st.session_state.active_game = active_selection
    st.session_state.attempts = 0
    st.session_state.game_over = False
    st.session_state.feedback = {}
    st.session_state.start_time = None  
    st.session_state.time_expired = False
    # Lightning mode state variables
    st.session_state.lt_started = False
    st.session_state.lt_chosen_metrics = []
    st.session_state.lt_max_questions = 30  
    st.session_state.lt_correct = 0
    st.session_state.lt_total = 0
    st.session_state.lt_current_q = None
    st.session_state.lt_last_feedback = ""
    # HOF Sprint state variables
    st.session_state.hof_started = False
    st.session_state.hof_duration_mins = 5
    st.session_state.hof_correct_guesses = []
    # Player ID Lightning state variables
    st.session_state.pid_started = False
    st.session_state.pid_max_questions = 15
    st.session_state.pid_correct = 0
    st.session_state.pid_total = 0
    st.session_state.pid_queue = []
    st.session_state.pid_current_q = None
    st.session_state.pid_last_feedback = ""
    # Mystery Roster state variables
    st.session_state.mr_started = False
    st.session_state.mr_decade = None
    st.session_state.mr_target = None
    st.session_state.mr_solved = False
    st.session_state.mr_final_score = None
    st.session_state.mr_last_feedback = ""
    st.session_state.mr_forced_reveals = 0
    # Roster Recall Lightning state variables
    st.session_state.rr_started = False
    st.session_state.rr_decade = None
    st.session_state.rr_target = None
    st.session_state.rr_correct_guesses = []
    st.session_state.rr_last_feedback = ""


# ==========================================
# 3. GLOBAL ENCAPSULATED TIMER FRAME
# ==========================================
if active_selection != "🏠 HOME SCREEN":
    if st.button("🏡 Return to Home Screen", key="global_home_btn"):
        st.session_state.nav_state = "🏠 HOME SCREEN"
        st.rerun()
        
    st.title(f"🏆 {active_selection}")

    # --- Authoritative expiry check ---
    # This runs inline as part of the normal script (every real rerun, e.g. each
    # guess submission) instead of on an independent background schedule, so it
    # can never collide with a form submission mid-render.
    remaining = 0
    if st.session_state.start_time is not None and not st.session_state.game_over:
        elapsed = time.time() - st.session_state.start_time
        if active_selection == "🏛️ HOF NAMING SPRINT":
            max_seconds = st.session_state.hof_duration_mins * 60
        elif active_selection == "🕵️ MYSTERY ROSTER":
            max_seconds = 300
        elif active_selection == "📋 ROSTER RECALL LIGHTNING":
            max_seconds = 120
        else:
            max_seconds = 420
        remaining = max(0, max_seconds - int(elapsed))
        if remaining <= 0:
            st.session_state.time_expired = True
            st.session_state.game_over = True

    # --- Cosmetic ticking clock ---
    # Pure client-side JavaScript: it never calls back into Streamlit, so it
    # can't trigger or collide with a rerun. It's reseeded with the accurate
    # server-side "remaining" value every time a real rerun happens anyway.
    if st.session_state.game_over and st.session_state.time_expired:
        st.error("⏰ TIME EXPIRED! Check your final stats below.")
    elif st.session_state.start_time is not None and not st.session_state.game_over:
        st.components.v1.html(f"""
            <div style="font-size:1.05rem;font-weight:600;color:#c0392b;
                 padding:0.55rem 1rem;border:1px solid #f5b7b1;border-radius:0.5rem;
                 background:#fdecea;font-family:inherit;display:inline-block;">
                ⏱️ TIME REMAINING: <span id="tt"></span>
            </div>
            <script>
                let remaining = {remaining};
                const el = document.getElementById("tt");
                function render() {{
                    const m = Math.floor(remaining / 60);
                    const s = String(remaining % 60).padStart(2, "0");
                    el.textContent = m + ":" + s;
                }}
                render();
                const iv = setInterval(() => {{
                    remaining = Math.max(0, remaining - 1);
                    render();
                    if (remaining <= 0) clearInterval(iv);
                }}, 1000);
            </script>
        """, height=48)

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
        st.warning("💪 **SOLID EFFORT!** Nice work! You passed.")
    else:
        st.error("🧱 **AIRBALL!** Tough game. Hit the tape, practice, and try again!")

# ==========================================
# BRANCH A: THE WELCOME HOME SCREEN
# ==========================================
if active_selection == "🏠 HOME SCREEN":
    st.title("🏀 Welcome to the NBA Complete Trivia Arena!")
    st.markdown("Test your historical basketball knowledge across decades of league records. Launch an interactive game mode right from this dashboard or use the left navigation sidebar.")
    
    st.write("---")
    st.write("## 🕹️ Select Your Way to Play")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Lightning Modes & Sprints")
        st.write("""
        * **Rapid Fire Blitz:** Custom-filter trivia pools to face a quick 25-50 random card session.
        * **Naismith HOF Sprint:** Race against a 3, 5, 7, or 9-minute buzzer to text-input as many Hall of Fame players as you can recall with smart typo leniency!
        * **Player ID Lightning:** We show you a player's career stat line, teams, and accolades — you type the name before time's up!
        * **Mystery Roster:** Pick a decade — we reveal a real team's roster one player at a time, bench guys first. Guess the team & year before the clock runs out!
        * **Roster Recall Lightning:** Pick a decade — we lock in a real team and year. Name as many players from that roster as you can in 2 minutes!
        """)
        s_btn1, s_btn2, s_btn3 = st.columns(3)
        with s_btn1:
            if st.button("⚡ Launch Lightning", use_container_width=True):
                st.session_state.nav_state = "⚡ LIGHTNING RAPID FIRE"
                st.rerun()
        with s_btn2:
            if st.button("🏛️ Launch HOF Sprint", use_container_width=True):
                st.session_state.nav_state = "🏛️ HOF NAMING SPRINT"
                st.rerun()
        with s_btn3:
            if st.button("🔍 Launch Player ID", use_container_width=True):
                st.session_state.nav_state = "🔍 PLAYER ID LIGHTNING"
                st.rerun()
        s_btn4, s_btn5 = st.columns(2)
        with s_btn4:
            if st.button("🕵️ Launch Mystery Roster", use_container_width=True):
                st.session_state.nav_state = "🕵️ MYSTERY ROSTER"
                st.rerun()
        with s_btn5:
            if st.button("📋 Launch Roster Recall", use_container_width=True):
                st.session_state.nav_state = "📋 ROSTER RECALL LIGHTNING"
                st.rerun()
        
    with col2:
        st.subheader("📋 Chronological Timeline Lists")
        st.write("""
        * **The Vibe:** Complete, deep-history grid charts.
        * **The Rules:** Select an individual category line. Fill out the multi-column historical grid table and submit your answers all at once.
        * **The Catch:** You get **3 attempts** to fix errors before answers reveal, capped by a **7-minute timer**.
        """)
        
    st.write("---")
    st.write("### 📋 Launch a Historical Timeline Category List:")
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        if st.button("🏅 Regular Season MVP", use_container_width=True):
            st.session_state.nav_state = "NBA MVP"; st.rerun()
        if st.button("🥇 Finals MVP", use_container_width=True):
            st.session_state.nav_state = "NBA Finals MVP"; st.rerun()
        if st.button("🛡️ Defensive Player (DPOY)", use_container_width=True):
            st.session_state.nav_state = "NBA defensive player of the year"; st.rerun()
    with b_col2:
        if st.button("🏆 Finals Champion", use_container_width=True):
            st.session_state.nav_state = "NBA finals winner"; st.rerun()
        if st.button("🥈 Finals Runner Up", use_container_width=True):
            st.session_state.nav_state = "NBA finals runner up"; st.rerun()
        if st.button("👶 Rookie of the Year", use_container_width=True):
            st.session_state.nav_state = "NBA Rookie of the year"; st.rerun()
    with b_col3:
        if st.button("🎯 Scoring Leader (PPG)", use_container_width=True):
            st.session_state.nav_state = "NBA Scoring leader"; st.rerun()
        if st.button("🪄 Assists Leader (APG)", use_container_width=True):
            st.session_state.nav_state = "NBA Assists leader"; st.rerun()
        if st.button("🪂 Rebound Leader (RPG)", use_container_width=True):
            st.session_state.nav_state = "NBA Rebound leader"; st.rerun()

# ==========================================
# BRANCH B: LIGHTNING RAPID FIRE GAME LOOP
# ==========================================
elif active_selection == "⚡ LIGHTNING RAPID FIRE":
    if not st.session_state.lt_started:
        st.write("### ⚙️ Configure Your Blitz Round")
        st.markdown("Choose your custom pools and length limit below. The 7-minute timer will not start until you press the launch button.")
        
        available_metrics = [k for k in game_modes.keys() if k not in ["⚡ LIGHTNING RAPID FIRE", "🏠 HOME SCREEN", "🏛️ HOF NAMING SPRINT", "🔍 PLAYER ID LIGHTNING", "🕵️ MYSTERY ROSTER", "📋 ROSTER RECALL LIGHTNING"]]
        chosen_metrics = st.multiselect("Metrics to include:", options=available_metrics, default=available_metrics)
        chosen_limit = st.selectbox("Number of questions for this round:", options=[25, 30, 40, 50], index=1)
        
        start_blitz = st.button("🚀 Start Blitz Game", disabled=len(chosen_metrics) == 0)
        if start_blitz:
            st.session_state.lt_chosen_metrics = chosen_metrics
            st.session_state.lt_max_questions = chosen_limit
            st.session_state.lt_started = True
            st.session_state.start_time = time.time()  
            st.rerun()
    else:
        if st.session_state.lt_total >= st.session_state.lt_max_questions:
            st.session_state.game_over = True

        if st.session_state.game_over:
            if st.session_state.lt_total >= st.session_state.lt_max_questions and not st.session_state.time_expired:
                st.success(f"🎯 **Completed all {st.session_state.lt_max_questions} questions!** Check your final stats:")
            render_grading_message(st.session_state.lt_correct, st.session_state.lt_total)
        else:
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

            # Form submit clears automatically without manual rerun conflicts
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
# BRANCH B2: PLAYER ID LIGHTNING
# ==========================================
elif active_selection == "🔍 PLAYER ID LIGHTNING":
    if player_meta_df.empty:
        st.warning("⚠️ 'nba_player_metadata.csv' not found. Run the updated build_nba_data.py scraper (scrape_player_metadata step) to generate it, then reload the app.")
    elif not st.session_state.pid_started:
        st.write("### 🔍 Player ID Lightning")
        st.markdown("We'll show you a player's career stat line, teams, and accolades — no name attached. Type who you think it is before time runs out!")

        chosen_limit = st.selectbox("Number of players to identify:", options=[10, 15, 20, 25], index=1)

        if st.button("🚀 Start Player ID Round", use_container_width=True):
            pool_size = len(player_meta_df)
            k = min(chosen_limit, pool_size)
            st.session_state.pid_queue = random.sample(range(pool_size), k=k)
            st.session_state.pid_max_questions = k
            st.session_state.pid_started = True
            st.session_state.pid_correct = 0
            st.session_state.pid_total = 0
            st.session_state.pid_current_q = None
            st.session_state.pid_last_feedback = ""
            st.session_state.start_time = time.time()
            st.rerun()
    else:
        if st.session_state.pid_total >= st.session_state.pid_max_questions:
            st.session_state.game_over = True

        if st.session_state.game_over:
            render_grading_message(st.session_state.pid_correct, st.session_state.pid_total)
        else:
            if st.session_state.pid_current_q is None:
                next_idx = st.session_state.pid_queue.pop(0)
                st.session_state.pid_current_q = player_meta_df.iloc[next_idx].to_dict()

            q = st.session_state.pid_current_q
            st.subheader(f"Player {st.session_state.pid_total + 1} of {st.session_state.pid_max_questions}")

            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.markdown(f"**Position:** {q.get('Position', 'N/A')}")
                st.markdown(f"**Years Active:** {q.get('Years_Active', 'N/A')}")
                st.markdown(f"**Teams:** {q.get('Teams', 'N/A')}")
            with info_col2:
                st.markdown(f"**Career PPG:** {q.get('PPG', 'N/A')}")
                st.markdown(f"**Career RPG:** {q.get('RPG', 'N/A')}")
                st.markdown(f"**Career APG:** {q.get('APG', 'N/A')}")

            accolade_bits = []
            for label, key in [("MVP", "MVP_Count"), ("DPOY", "DPOY_Count"), ("Finals MVP", "FMVP_Count"), ("ROY", "ROY_Count")]:
                count = q.get(key, 0)
                try:
                    count = int(count)
                except (TypeError, ValueError):
                    count = 0
                if count > 0:
                    accolade_bits.append(f"{count}x {label}")
            if str(q.get("Accolades", "")).strip() not in ("", "N/A", "nan"):
                accolade_bits.append(str(q["Accolades"]))
            if accolade_bits:
                st.info("🏅 " + " • ".join(accolade_bits))

            with st.form("pid_entry_form", clear_on_submit=True):
                user_guess = st.text_input("Who is this player?", placeholder="Type a full name and press enter...")
                submit_guess = st.form_submit_button("Submit Guess", use_container_width=True)

            if submit_guess and user_guess.strip() != "":
                raw_guess = user_guess.strip()
                best_match, score = process.extractOne(raw_guess, [q["Player"]])
                st.session_state.pid_total += 1
                if score >= 85:
                    st.session_state.pid_correct += 1
                    st.session_state.pid_last_feedback = f"✅ **Correct!** It was *{q['Player']}*."
                else:
                    st.session_state.pid_last_feedback = f"❌ **Incorrect.** You guessed *{raw_guess}*. It was *{q['Player']}*."
                st.session_state.pid_current_q = None
                st.rerun()

            if st.session_state.pid_last_feedback:
                st.markdown(st.session_state.pid_last_feedback)

# ==========================================
# BRANCH B3: MYSTERY ROSTER (DECADE GUESSING GAME)
# ==========================================
elif active_selection == "🕵️ MYSTERY ROSTER":
    if decade_rosters_df.empty:
        st.warning("⚠️ 'nba_decade_rosters.csv' not found. Run the updated build_nba_data.py scraper (scrape_decade_rosters step) to generate it, then reload the app.")
    elif not st.session_state.mr_started:
        st.write("### 🕵️ Mystery Roster")
        st.markdown("""
        Pick a decade. We'll reveal a real NBA team's roster one player at a time —
        starting with the deepest bench player (lowest minutes per game) and working
        up to the stars — one new name every 30 seconds, across a 5-minute clock.
        You can also force the next name early at any time if you want more clues
        sooner. Guess the **team and year** as many times as you like. The fewer
        players revealed when you nail it, the more points you keep (starts at 20,
        minus 1 for every player revealed after the first).
        """)

        available_decades = sorted(decade_rosters_df['Decade'].unique().tolist())
        chosen_decade = st.selectbox("Choose a decade:", options=available_decades)

        if st.button("🚀 Start Mystery Roster", use_container_width=True):
            pool = decade_rosters_df[decade_rosters_df['Decade'] == chosen_decade]
            target_row = pool.sample(1).iloc[0]
            st.session_state.mr_decade = chosen_decade
            st.session_state.mr_target = {
                "Team": target_row["Team"],
                "TeamFull": target_row["TeamFull"],
                "Year": int(target_row["Year"]),
                "PlayerOrder": [p for p in str(target_row["PlayerOrder"]).split("|") if p],
            }
            st.session_state.mr_started = True
            st.session_state.mr_solved = False
            st.session_state.mr_final_score = None
            st.session_state.mr_last_feedback = ""
            st.session_state.mr_forced_reveals = 0
            st.session_state.start_time = time.time()
            st.rerun()
    else:
        target = st.session_state.mr_target
        roster = target["PlayerOrder"]
        elapsed = time.time() - st.session_state.start_time
        time_based_count = min(len(roster), 1 + int(elapsed // 30)) if roster else 0
        revealed_count = max(time_based_count, min(len(roster), st.session_state.mr_forced_reveals)) if roster else 0

        if st.session_state.game_over:
            st.write("---")
            st.subheader("🏁 Mystery Roster Result")
            if st.session_state.mr_solved:
                score = st.session_state.mr_final_score
                st.info(f"**You solved it!** It was the **{target['Year']} {target['TeamFull']}**.")
                st.metric(label="Points Earned", value=f"{score} / 20")
                if score >= 18:
                    st.balloons()
                    st.success("👑 **MASTER DETECTIVE!** You barely needed a clue.")
                elif score >= 12:
                    st.success("🔍 **SHARP EYE!** Great read on that roster.")
                elif score >= 6:
                    st.warning("👀 **DECENT READ.** You got there eventually.")
                else:
                    st.error("🐢 **JUST IN TIME.** Down to the wire, but you got it!")
            else:
                st.error(f"⏰ **Time's up!** The answer was the **{target['Year']} {target['TeamFull']}**.")
                st.metric(label="Points Earned", value="0 / 20")

            with st.expander("👀 Full roster reveal order:"):
                st.write(", ".join(roster) if roster else "No roster data.")
        else:
            st.write(f"### Decade: {st.session_state.mr_decade}")
            st.write(f"**Players revealed so far ({revealed_count}):**")
            st.info(", ".join(roster[:revealed_count]) if roster else "No roster data available.")
            st.caption(f"Current potential score if correct: **{max(0, 20 - (revealed_count - 1))} / 20**")

            reveal_col1, reveal_col2 = st.columns(2)
            with reveal_col1:
                if st.button("🔄 Check for New Reveal", use_container_width=True, help="Free — just syncs the display to however much time has actually passed."):
                    st.rerun()
            with reveal_col2:
                next_reveal_possible = revealed_count < len(roster)
                if st.button("👀 Reveal Next Player Now (-1 pt)", use_container_width=True, disabled=not next_reveal_possible, help="Forces the next player to show immediately, even if 30 seconds haven't passed."):
                    st.session_state.mr_forced_reveals = min(len(roster), revealed_count + 1)
                    st.rerun()

            candidate_years = sorted(decade_rosters_df[decade_rosters_df['Decade'] == st.session_state.mr_decade]['Year'].unique().tolist())

            with st.form("mr_guess_form"):
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    guess_year = st.selectbox("Year:", options=candidate_years, index=None, placeholder="Choose year...")
                with g_col2:
                    mr_team_options = sorted(decade_rosters_df['Team'].dropna().unique().tolist())
                    guess_team = st.selectbox("Team:", options=mr_team_options, index=None, placeholder="Choose team...")
                submit_guess = st.form_submit_button("Submit Guess", use_container_width=True)

            if submit_guess:
                if guess_year == target["Year"] and guess_team == target["Team"]:
                    st.session_state.mr_solved = True
                    st.session_state.mr_final_score = max(0, 20 - (revealed_count - 1))
                    st.session_state.game_over = True
                    st.rerun()
                else:
                    st.session_state.mr_last_feedback = "❌ Not quite — keep watching for more clues!"

            if st.session_state.mr_last_feedback:
                st.markdown(st.session_state.mr_last_feedback)

# ==========================================
# BRANCH B4: ROSTER RECALL LIGHTNING
# ==========================================
elif active_selection == "📋 ROSTER RECALL LIGHTNING":
    if decade_rosters_df.empty:
        st.warning("⚠️ 'nba_decade_rosters.csv' not found. Run the updated build_nba_data.py scraper (scrape_decade_rosters step) to generate it, then reload the app.")
    elif not st.session_state.rr_started:
        st.write("### 📋 Roster Recall Lightning")
        st.markdown("""
        Pick a decade. We'll lock in a real NBA team from a specific season — you
        won't be told which players are on it. You have **2 minutes** to name as
        many players from that roster as you can, with typo-tolerant matching.
        1 point per correct, unique name. Unlimited guesses — go fast!
        """)

        available_decades = sorted(decade_rosters_df['Decade'].unique().tolist())
        chosen_decade = st.selectbox("Choose a decade:", options=available_decades, key="rr_decade_select")

        if st.button("🚀 Start Roster Recall", use_container_width=True):
            pool = decade_rosters_df[decade_rosters_df['Decade'] == chosen_decade]
            target_row = pool.sample(1).iloc[0]
            st.session_state.rr_decade = chosen_decade
            st.session_state.rr_target = {
                "Team": target_row["Team"],
                "TeamFull": target_row["TeamFull"],
                "Year": int(target_row["Year"]),
                "PlayerOrder": [p for p in str(target_row["PlayerOrder"]).split("|") if p],
            }
            st.session_state.rr_started = True
            st.session_state.rr_correct_guesses = []
            st.session_state.rr_last_feedback = ""
            st.session_state.game_over = False
            st.session_state.time_expired = False
            st.session_state.start_time = time.time()
            st.rerun()
    else:
        target = st.session_state.rr_target
        roster = target["PlayerOrder"]

        if st.session_state.game_over:
            st.write("---")
            st.subheader("🏁 Roster Recall Result")
            score = len(st.session_state.rr_correct_guesses)
            total = len(roster)
            st.info(f"The roster was the **{target['Year']} {target['TeamFull']}**.")
            st.metric(label="Players Named", value=f"{score} / {total}")

            pct = (score / total) if total else 0
            if pct >= 0.75:
                st.balloons()
                st.success("👑 **ROSTER MASTER!** Incredible recall on that squad.")
            elif pct >= 0.5:
                st.success("🔥 **STRONG SQUAD KNOWLEDGE!** Nicely done.")
            elif pct >= 0.25:
                st.warning("👍 **DECENT EFFORT.** You got a good chunk of them.")
            else:
                st.error("🧊 **COLD START.** That roster stumped you a bit!")

            missed = [p for p in roster if p not in st.session_state.rr_correct_guesses]
            with st.expander(f"👀 Full roster ({total} players):"):
                st.write("**✅ You got:** " + (", ".join(st.session_state.rr_correct_guesses) if st.session_state.rr_correct_guesses else "None"))
                st.write("**❌ Missed:** " + (", ".join(missed) if missed else "None — you got them all!"))
        else:
            st.write(f"### Decade: {st.session_state.rr_decade}")
            st.write(f"**Score: {len(st.session_state.rr_correct_guesses)} player(s) named**")
            if st.session_state.rr_correct_guesses:
                st.info(", ".join(st.session_state.rr_correct_guesses))

            with st.form("rr_entry_form", clear_on_submit=True):
                user_guess = st.text_input("Name a player on this roster:", placeholder="Type a full name and press enter...")
                submit_guess = st.form_submit_button("Submit Name", use_container_width=True)

            if submit_guess and user_guess.strip() != "":
                raw_guess = user_guess.strip()
                best_match, match_score = process.extractOne(raw_guess, roster) if roster else (None, 0)
                if best_match and match_score >= 85:
                    if best_match not in st.session_state.rr_correct_guesses:
                        st.session_state.rr_correct_guesses.append(best_match)
                        st.toast(f"✅ {best_match} ({match_score}% match)", icon="🔥")
                        st.session_state.rr_last_feedback = f"✅ **{best_match}** confirmed!"
                    else:
                        st.toast(f"⚠️ You already named {best_match}!", icon="👀")
                        st.session_state.rr_last_feedback = f"⚠️ You already named **{best_match}**."
                else:
                    st.toast(f"❌ '{raw_guess}' wasn't on this roster.", icon="🧱")
                    st.session_state.rr_last_feedback = f"❌ '{raw_guess}' wasn't on this roster."

            if st.session_state.rr_last_feedback:
                st.markdown(st.session_state.rr_last_feedback)

# ==========================================
# BRANCH C: HALL OF FAME NAMING SPRINT
# ==========================================
elif active_selection == "🏛️ HOF NAMING SPRINT":
    if not st.session_state.hof_started:
        st.write("### 🏛️ Naismith Hall of Fame Naming Sprint")
        st.markdown("How many NBA Hall of Fame players can you name before the buzzer sounds? Type names one by one. Spelling counts, but our engine is smart enough to accept close typos!")
        
        chosen_mins = st.selectbox("Select Sprint Duration:", options=[3, 5, 7, 9], index=1)
        
        if st.button("🏁 Start Sprint Round", use_container_width=True):
            st.session_state.hof_duration_mins = chosen_mins
            st.session_state.hof_started = True
            st.session_state.start_time = time.time()
            st.session_state.hof_correct_guesses = []
            st.rerun()
    else:
        elapsed = time.time() - st.session_state.start_time
        max_sec = st.session_state.hof_duration_mins * 60
        if elapsed >= max_sec:
            st.session_state.game_over = True

        if st.session_state.game_over:
            st.write("---")
            st.subheader("🏁 Sprint Complete!")
            count = len(st.session_state.hof_correct_guesses)
            mins_selected = st.session_state.hof_duration_mins
            
            gpm = round(count / mins_selected, 1) 
            st.info(f"**Final Score:** {count} Players Named in {mins_selected} Minutes")
            st.metric(label="Your Typing Velocity (Guesses Per Minute)", value=f"{gpm} GPM")
            
            if gpm >= 12.0:
                st.balloons()
                st.success("👑 **STATISTICAL SAVANT!** Your pace is historic! Pure elite recall memory.")
            elif gpm >= 7.0:
                st.success("🔥 **ALL-STAR PACE!** Outstanding speed! Your basketball knowledge is deep.")
            elif gpm >= 3.5:
                st.warning("💪 **SOLID ROTATION PLAYER!** Respectable hustle! You maintained a steady pace.")
            else:
                st.error("🧱 **BENCHWARMER COMPOSURE.** Oof, a bit sluggish out there. Review the tape and try a faster game!")

            with st.expander("👀 Review the ones you missed:"):
                missed = [p for p in hof_master_list if p not in st.session_state.hof_correct_guesses]
                st.write(", ".join(sorted(missed)))
        else:
            st.write(f"### Score: **{len(st.session_state.hof_correct_guesses)}** Players Named")
            
            # Form actions update state variables locally; manual st.rerun removed to ensure form stability
            with st.form("hof_entry_form", clear_on_submit=True):
                user_input = st.text_input("Type a player name and press enter:", placeholder="e.g. Larry Bird, Magic Johnson...")
                submit_name = st.form_submit_button("Submit Name", use_container_width=True)
                
            if submit_name and user_input.strip() != "":
                raw_guess = user_input.strip()
                best_match, score = process.extractOne(raw_guess, hof_master_list)
                
                if score >= 85:
                    if best_match not in st.session_state.hof_correct_guesses:
                        st.session_state.hof_correct_guesses.append(best_match)
                        st.toast(f"✅ Confirmed: {best_match} ({score}% match)", icon="🔥")
                    else:
                        st.toast(f"⚠️ You already named {best_match}!", icon="👀")
                else:
                    st.toast(f"❌ '{raw_guess}' didn't match any HOF players.", icon="🧱")

            if st.session_state.hof_correct_guesses:
                st.write("### 📝 Your Confirmed Hall of Famers:")
                st.write(", ".join(st.session_state.hof_correct_guesses))

# ==========================================
# BRANCH D: REGULAR TIMELINE LIST GAME MODES
# ==========================================
else:
    # Aggressive JavaScript timed scroll override execution block
    st.components.v1.html(
        "<script>setTimeout(function(){window.parent.scrollTo({top:0,left:0,behavior:'instant'});window.scrollTo({top:0,left:0,behavior:'instant'});document.documentElement.scrollTop=0;document.body.scrollTop=0;},100);</script>", 
        height=0, width=0
    )

    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()

    local_start_year = game_cfg["start_year"]
    local_target_col = str(game_cfg["col"])
    local_game_type = game_cfg["type"]
    local_limited = game_cfg.get("limited_options", False)

    filtered_df = df[df['Year'] >= local_start_year].sort_values(by="Year", ascending=False)
    
    with st.form("timeline_form"):
        st.write(f"### Attempt {st.session_state.attempts}/3")
        user_guesses = {}
        
        # Split layout data items into rows containing up to 4 columns side by side
        rows_data = [filtered_df[i:i + 4] for i in range(0, len(filtered_df), 4)]
        
        for row_chunk in rows_data:
            cols = st.columns(4)
            for index, (idx, row) in enumerate(row_chunk.iterrows()):
                year = int(row['Year'])
                if local_limited:
                    past_winners = df[(df['Year'] <= year) & (df['Year'] >= local_start_year)].sort_values(by="Year", ascending=False)
                    dropdown_options = sorted(list(past_winners[local_target_col].astype(str).unique()[:5]))
                else:
                    dropdown_options = global_teams if local_game_type == "team" else global_players
                
                with cols[index]:
                    guess = st.selectbox(
                        f"Year {year}", options=dropdown_options, index=None,
                        placeholder="Choose...", key=f"{active_selection}_{year}_{idx}", disabled=st.session_state.game_over
                    )
                    user_guesses[year] = guess or ""
            
        st.write("---")
        submit_btn = st.form_submit_button("Submit Entire List", disabled=st.session_state.game_over, use_container_width=True)

    if submit_btn:
        st.session_state.attempts += 1
        results = {}
        all_correct = True
        
        for idx, row in filtered_df.iterrows():
            year = int(row['Year'])
            actual = str(row[local_target_col]).lower().strip()
            user_val = str(user_guesses.get(year, "")).lower().strip()
            is_correct = (user_val == actual)
            results[year] = is_correct
            if not is_correct: all_correct = False
                
        st.session_state.feedback = results
        if all_correct or st.session_state.attempts >= 3:
            st.session_state.game_over = True
        st.rerun()

    # Score Overview Render Grid
    if st.session_state.feedback:
        total_items = len(st.session_state.feedback)
        correct_count = sum(1 for v in st.session_state.feedback.values() if v is True)
        
        if st.session_state.game_over:
            render_grading_message(correct_count, total_items)
        else:
            st.write(f"### 📝 Current List Status (Attempts Remaining: {3 - st.session_state.attempts}):")
            
        feedback_rows = [filtered_df[i:i + 4] for i in range(0, len(filtered_df), 4)]
        for f_row in feedback_rows:
            f_cols = st.columns(4)
            for f_index, (idx, row) in enumerate(f_row.iterrows()):
                year = int(row['Year'])
                actual = str(row[local_target_col])
                is_ok = st.session_state.feedback.get(year, False)
                with f_cols[f_index]:
                    if is_ok:
                        st.markdown(f"✅ **{year}:** Correct")
                    else:
                        reveal = f" *({actual})*".replace("N/A", "No Award") if st.session_state.game_over else ""
                        st.markdown(f"❌ **{year}:** {reveal}")
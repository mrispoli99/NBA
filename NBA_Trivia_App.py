import pandas as pd
import numpy as np
from data_loader import get_era_baselines

def calculate_real_pdi(df):
    """
    Applies multi-variable adjustments (Pace, Scale, Competition Concentration, 
    Positional Perimeter Stress, and Continuous Teammate Burden) to a seasonal group.
    """
    if df.empty:
        return df
        
    year = int(df['Season'].iloc[0])
    baselines = get_era_baselines(year)
    
    n_teams = baselines['historical_teams']
    league_base = baselines['league_baseline']
    kinematic_base = baselines['kinematic_baseline']
    
    # 1. Base Performance Layer: Seasonal Win Share Z-Score
    df['dominance_z'] = (df['WS'] - df['WS'].mean()) / df['WS'].std()
    
    pdi_scores = []
    rsd_list = []
    psp_list = []
    scb_list = []
    
    for idx, row in df.iterrows():
        # A. Strength of Competition (Concentration of 20+ PPG Buckets across the league)
        buckets_per_team = row['League_20_PPG_Count'] / n_teams
        competition_factor = 1.0 + (buckets_per_team * 0.15)
        
        # B. Spacing & Positional Perimeter Stress Engine
        # Handles the paradox of physical clog down low vs modern hunting/switching on the perimeter
        if row['3P%'] < 0.10:
            # PRE-3PT LINE / EARLY ERA: High structural physical congestion in the paint
            spacing_friction = 1.15
        else:
            # MODERN SPACING ERA: Floor weaponization and perimeter recovery stress
            base_spacing_stress = 1.0 + (row['3P%'] * 0.4) 
            
            # Positional Stress Modifier: Frontcourt players (C and PF) get hit with a premium
            # because they are forced out of the paint to defend pick-and-rolls out to the logo.
            player_pos = str(row['Pos']).upper()
            if 'C' in player_pos or 'F' in player_pos:
                position_stress_multiplier = 1.10  # 10% premium for defensive perimeter range
            else:
                position_stress_multiplier = 1.02  # Small guard chasing premium
                
            spacing_friction = base_spacing_stress * position_stress_multiplier
        
        # Combine macro filters into the Regular Season Baseline
        era_multiplier = league_base * kinematic_base * (np.log(n_teams) / np.log(30)) * competition_factor * spacing_friction
        rsd = row['dominance_z'] * era_multiplier
        
        # C. Playoff Bracket Format Friction
        if row['MP'] > 2400 and row['WS'] > 8.0:
            rounds = 4.0 if year >= 2003 else (3.5 if year >= 1984 else (2.5 if year >= 1970 else 1.5))
            psp = 0.6 * (rounds * (np.log(n_teams) / np.log(8)))
        else:
            psp = 0
            
        # D. Continuous Teammate Efficiency Gradient (The Support Fix)
        # Centered exactly at a league-average 15.0 PER. No sudden cliffs.
        # Every point below 15 rewards a carrying job; every point above applies a luxury deflation.
        scb = 1.0 + ((15.0 - row['Core_Support_PER']) * 0.04)
        scb = np.clip(scb, 0.75, 1.30)  # Capped at a maximum 30% modifier swinging either way
        
        # Combine all structural layers
        raw_pdi = (rsd + psp) * scb
        pdi_scores.append(max(0, raw_pdi))
        rsd_list.append(rsd)
        psp_list.append(psp)
        scb_list.append(scb)
        
    df['raw_pdi'] = pdi_scores
    df['Regular_Season_Difficulty_Base'] = rsd_list
    df['Playoff_Format_Friction'] = psp_list
    df['Supporting_Cast_Modifier'] = scb_list
    df['Total_League_Teams'] = n_teams
    
    return df.sort_values(by='WS', ascending=False).head(50)

def load_all_real_seasons():
    """
    Loads the full pre-scraped 1950-2026 archive from the local CSV,
    processes seasonal baselines, and applies a true global max 0-100 normalization.
    """
    try:
        raw_df = pd.read_csv("nba_raw_archive.csv")
    except FileNotFoundError:
        print("Error: 'nba_raw_archive.csv' not found. Please run build_database.py first.")
        return pd.DataFrame()
        
    processed_seasons = []
    for season, group in raw_df.groupby('Season'):
        processed_group = calculate_real_pdi(group)
        processed_seasons.append(processed_group)
        
    full_dataset = pd.concat(processed_seasons, ignore_index=True)
    
    # Global Max Normalization (Standardized Anchor across history)
    global_max = full_dataset['raw_pdi'].max()
    full_dataset['pdi_final'] = (full_dataset['raw_pdi'] / global_max) * 100
    
    return full_dataset
import streamlit as st
import pandas as pd
import numpy as np
import warnings
import nfl_data_py as nfl

warnings.filterwarnings("ignore")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="NFL Defender Attention & Impact",
    layout="wide"
)

# --------------------------------------------------
# DATA LOAD (relative paths)
# --------------------------------------------------
@st.cache_data
def load_data():
    """Load all required data with caching"""
    df = pd.read_csv("defender_attention_all.csv")
    nfl_players = nfl.import_players()
    nfl_teams = nfl.import_team_desc()
    df_detailed_results = pd.read_csv("intervention_detailed_results.csv")
    return df, nfl_players, nfl_teams, df_detailed_results

df, nfl_players, nfl_teams, df_detailed_results = load_data()

# --------------------------------------------------
# FUNCTION DEFINITION
# --------------------------------------------------

def create_player_attention_table(df, nfl_players, nfl_teams, df_detailed_results=None):
    """
    Create an interactive player attention table with Streamlit
    
    Parameters:
    -----------
    df : DataFrame
        Contains player attention metrics with columns:
        ['nfl_id', 'total_attention', 'avg_attention', 'max_attention', 
         'std_attention', 'median_attention', 'play_count', 'frame_count', 'high_attention_pct']
    
    nfl_players : DataFrame
        Contains player info with 'nfl_id' and other player details
    
    nfl_teams : DataFrame
        Contains team info with 'team_abbr' and color/logo information
    
    df_detailed_results : DataFrame, optional
        Contains intervention details with columns:
        ['game_id', 'play_id', 'defender_nfl_id', 'intervention_type', 
         'original_prediction', 'counterfactual_prediction', 'impact_score']
    """
    
    # ===== CRITICAL: Ensure nfl_id is consistent dtype across all dataframes =====
    df = df.copy()
    nfl_players = nfl_players.copy()
    nfl_teams = nfl_teams.copy()
    
    df['nfl_id'] = df['nfl_id'].astype(str)
    nfl_players['nfl_id'] = nfl_players['nfl_id'].astype(str)
    nfl_teams['team_abbr'] = nfl_teams['team_abbr'].astype(str)
    
    # Position grouping mapping
    position_groups = {
        'LB': ['LB', 'ILB', 'MLB', 'OLB'],
        'CB': ['CB', 'DB'],
        'S': ['S', 'SAF']
    }
    
    # Merge with player information
    df_with_players = df.merge(
        nfl_players[['nfl_id', 'display_name', 'position', 'latest_team', 'headshot']],
        on='nfl_id',
        how='left'
    )
    
    # Merge with team information
    df_with_players = df_with_players.merge(
        nfl_teams[['team_abbr', 'team_color', 'team_color2', 'team_logo_squared', 'team_wordmark']],
        left_on='latest_team',
        right_on='team_abbr',
        how='left'
    )
    
    # Add position group column
    def get_position_group(pos):
        if pd.isna(pos):
            return 'Other'
        for group, positions in position_groups.items():
            if any(p in str(pos) for p in positions):
                return group
        return 'Other'
    
    df_with_players['position_group'] = df_with_players['position'].apply(get_position_group)
    
    # Add intervention type counts if df_detailed_results is provided
    if df_detailed_results is not None and len(df_detailed_results) > 0:
        df_detailed_results_copy = df_detailed_results[df_detailed_results['intervention_type'] == 'REMOVAL'].copy()
        df_detailed_results_copy['defender_nfl_id'] = df_detailed_results_copy['defender_nfl_id'].astype(str)
        
        # Count interventions by type for each player and calculate average impact score
        removal_stats = df_detailed_results_copy.groupby('defender_nfl_id').agg(
            intervention_removal=('defender_nfl_id', 'size'),
            impact_removal=('impact_score', 'mean')).reset_index()
        
        df_with_players = df_with_players.merge(
            removal_stats, 
            left_on='nfl_id', 
            right_on='defender_nfl_id', 
            how='left')
        
        df_with_players['intervention_removal'] = df_with_players['intervention_removal'].fillna(0).astype(int)
        df_with_players['impact_removal'] = df_with_players['impact_removal'].fillna(0)
        
        # Get total intervention count
        total_interventions = df_detailed_results_copy.groupby('defender_nfl_id').size().reset_index(name='total_interventions')
        df_with_players = df_with_players.merge(
            total_interventions,
            left_on='nfl_id',
            right_on='defender_nfl_id',
            how='left'
        )
        df_with_players['total_interventions'] = df_with_players['total_interventions'].fillna(0).astype(int)
        
        # Drop duplicate defender_nfl_id column
        if 'defender_nfl_id' in df_with_players.columns:
            df_with_players = df_with_players.drop('defender_nfl_id', axis=1)
    
    # ===== Custom CSS Styling =====
    st.markdown("""
    <style>
        /* Main container styling */
        .metric-container {
            background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 100%);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #374151;
        }
        
        /* Table styling */
        .dataframe {
            width: 100% !important;
        }
        
        .dataframe th {
            background-color: #1f2937 !important;
            color: #e5e7eb !important;
            font-weight: 600 !important;
            text-align: center !important;
            padding: 10px !important;
            font-size: 11px !important;
        }
        
        .dataframe td {
            text-align: center !important;
            padding: 10px !important;
            color: #d1d5db !important;
        }
        
        .dataframe tr:hover {
            background-color: rgba(59, 130, 246, 0.1) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ===== Streamlit Layout =====
    
    # Title section
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%); border-radius: 10px; border: 1px solid rgba(59, 130, 246, 0.3); margin-bottom: 20px;'>
        <h1 style='margin: 0; background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 28px;'>🏈 Neural Network Attention & Impact Analysis</h1>
        <p style='color: #9ca3af; margin: 8px 0 0 0;'>Top Defenders in 2023</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        position_filter = st.selectbox(
            "Position",
            options=['All Positions', 'LB (Linebackers)', 'CB (Cornerbacks)', 'S (Safeties)', 'Other Positions'],
            index=0,
            key='position_filter'
        )
    
    with col2:
        plays_filter = st.selectbox(
            "Play Volume",
            options=['No Filter', '≥ 20% of Max Plays', '≥ 50% of Max Plays', '≥ 80% of Max Plays'],
            index=1,
            key='plays_filter'
        )
    
    with col3:
        sort_options = [
            'Avg Attention (High to Low)',
            'High Attention % (High to Low)',
            'Play Count (High to Low)',
            'Impact (High to Low)',
            'Player Name (A-Z)'
        ]
        sort_by = st.selectbox(
            "Sort by",
            options=sort_options,
            index=3,
            key='sort_by'
        )
    
    # ===== Apply Filters =====
    filtered_df = df_with_players.copy()
    
    # Position filter
    position_map = {
        'All Positions': 'all',
        'LB (Linebackers)': 'LB',
        'CB (Cornerbacks)': 'CB',
        'S (Safeties)': 'S',
        'Other Positions': 'Other'
    }
    
    pos_value = position_map.get(position_filter, 'all')
    if pos_value != 'all':
        filtered_df = filtered_df[filtered_df['position_group'] == pos_value]
    
    # Plays filter
    plays_map = {
        'No Filter': 0,
        '≥ 20% of Max Plays': 20,
        '≥ 50% of Max Plays': 50,
        '≥ 80% of Max Plays': 80
    }
    
    plays_value = plays_map.get(plays_filter, 0)
    if plays_value > 0:
        max_plays_in_filtered = filtered_df['play_count'].max() if len(filtered_df) > 0 else 1
        threshold = (plays_value / 100) * max_plays_in_filtered
        filtered_df = filtered_df[filtered_df['play_count'] >= threshold]
    
    # Sorting
    sort_map = {
        'Avg Attention (High to Low)': ('avg_attention', False),
        'High Attention % (High to Low)': ('high_attention_pct', False),
        'Play Count (High to Low)': ('play_count', False),
        'Impact (High to Low)': ('impact_removal', False),
        'Player Name (A-Z)': ('display_name', True)
    }
    
    sort_col, ascending = sort_map.get(sort_by, ('impact_removal', False))
    if sort_col in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_col, ascending=ascending)
    
    # Add rank
    filtered_df = filtered_df.copy()
    filtered_df['rank'] = range(1, len(filtered_df) + 1)
    
    # ===== Display Results =====
    
    # Results info
    st.markdown(f"""
    <div style='padding: 10px; margin-bottom: 20px; color: #9ca3af; font-size: 14px;'>
        Showing <span style='color: #3b82f6; font-weight: 600;'>{len(filtered_df)}</span> of <span style='color: #3b82f6; font-weight: 600;'>{len(df_with_players)}</span> players
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare display dataframe
    display_df = filtered_df[[
        'rank', 'display_name', 'position', 'latest_team', 'play_count',
        'high_attention_pct', 'avg_attention', 'impact_removal'
    ]].copy()
    
    display_df.columns = ['Rank', 'Player', 'Position', 'Team', 'Plays', 'High Attn %', 'Avg Attn', 'Impact']
    
    # Format numeric columns
    display_df['High Attn %'] = display_df['High Attn %'].apply(lambda x: f"{x:.1f}%")
    display_df['Avg Attn'] = display_df['Avg Attn'].apply(lambda x: f"{x*100:.1f}%")
    display_df['Impact'] = display_df['Impact'].apply(lambda x: f"{x*100:.1f}%")
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Rank': st.column_config.NumberColumn(width='small'),
            'Player': st.column_config.TextColumn(width='large'),
            'Position': st.column_config.TextColumn(width='small'),
            'Team': st.column_config.TextColumn(width='small'),
            'Plays': st.column_config.NumberColumn(width='small'),
            'High Attn %': st.column_config.TextColumn(width='small'),
            'Avg Attn': st.column_config.TextColumn(width='small'),
            'Impact': st.column_config.TextColumn(width='small')
        }
    )
    
    # ===== Legend Section =====
    st.markdown("---")
    st.markdown("### 📋 Intervention Type Definitions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **REMOVAL**  
        Removes defender from all frames  
        
        *What would happen if this defender wasn't on the field at all?*
        """)
    
    with col2:
        st.markdown("""
        **FREEZE**  
        Freezes defender at initial position  
        
        *What if this defender didn't react or pursue at all?*
        """)
    
    with col3:
        st.markdown("""
        **SLOWDOWN**  
        Reduces speed by 50%  
        
        *What if this defender was slower/less athletic?*
        """)
    
    with col4:
        st.markdown("""
        **MISDIRECTION**  
        Rotates velocity by 90 degrees  
        
        *What if this defender took a bad angle or was fooled?*
        """)
    
    return filtered_df


# --------------------------------------------------
# MAIN APP
# --------------------------------------------------

# Call the function to render the table
create_player_attention_table(df, nfl_players, nfl_teams, df_detailed_results)

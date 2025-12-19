import streamlit as st
import pandas as pd
from player_attention_table_streamlit import create_player_attention_table

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="NFL Player Attention Analysis",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Optional: Add custom CSS for the overall page
st.markdown("""
    <style>
        .main {
            background-color: #0f172a;
        }
        .stSelectbox > div > div {
            background-color: #1e293b;
        }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("🏈 NFL Player Attention Analysis Dashboard")
st.markdown("---")

# Load your data
# Replace these with your actual data loading methods
@st.cache_data
def load_data():
    """
    Load all required dataframes
    You should replace this with your actual data loading logic
    """
    # Example - replace with your actual file paths or data loading
    df = pd.read_csv('player_attention_metrics.csv')
    nfl_players = pd.read_csv('nfl_players.csv')
    nfl_teams = pd.read_csv('nfl_teams.csv')
    df_detailed_results = pd.read_csv('detailed_results.csv')
    
    return df, nfl_players, nfl_teams, df_detailed_results

# Load the data
try:
    df, nfl_players, nfl_teams, df_detailed_results = load_data()
    
    # Create the interactive table
    create_player_attention_table(df, nfl_players, nfl_teams, df_detailed_results)
    
except FileNotFoundError as e:
    st.error(f"Data files not found: {e}")
    st.info("Please ensure all required CSV files are in the correct directory.")
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.exception(e)

# Optional: Add footer or additional information
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #9ca3af; font-size: 12px; padding: 20px;'>
        NFL Player Attention Analysis | Data from 2023 Season
    </div>
""", unsafe_allow_html=True)

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
# DATA LOAD (YOUR EXACT PATHS)
# --------------------------------------------------
df = pd.read_csv("defender_attention_all.csv")
nfl_players = nfl.import_players()
nfl_teams = nfl.import_team_desc()
df_detailed_results = pd.read_csv("intervention_detailed_results.csv")

# --------------------------------------------------
# CORE TRANSFORM FUNCTION (UNCHANGED)
# --------------------------------------------------
def build_base_dataframe(df, nfl_players, nfl_teams, df_detailed_results):

    df = df.copy()
    nfl_players = nfl_players.copy()
    nfl_teams = nfl_teams.copy()

    df["nfl_id"] = df["nfl_id"].astype(str)
    nfl_players["nfl_id"] = nfl_players["nfl_id"].astype(str)
    nfl_teams["team_abbr"] = nfl_teams["team_abbr"].astype(str)

    position_groups = {
        "LB": ["LB", "ILB", "MLB", "OLB"],
        "CB": ["CB", "DB"],
        "S": ["S", "SAF"]
    }

    df = df.merge(
        nfl_players[["nfl_id", "display_name", "position", "latest_team", "headshot"]],
        on="nfl_id",
        how="left"
    )

    df = df.merge(
        nfl_teams[["team_abbr", "team_color", "team_color2"]],
        left_on="latest_team",
        right_on="team_abbr",
        how="left"
    )

    def get_position_group(pos):
        if pd.isna(pos):
            return "Other"
        for g, vals in position_groups.items():
            if any(v in str(pos) for v in vals):
                return g
        return "Other"

    df["position_group"] = df["position"].apply(get_position_group)

    # Removal-only intervention stats
    df_detailed_results = df_detailed_results[
        df_detailed_results["intervention_type"] == "removal"
    ].copy()

    df_detailed_results["defender_nfl_id"] = df_detailed_results["defender_nfl_id"].astype(str)

    removal_stats = (
        df_detailed_results
        .groupby("defender_nfl_id")
        .agg(
            intervention_removal=("defender_nfl_id", "size"),
            impact_removal=("impact_score", "mean")
        )
        .reset_index()
    )

    df = df.merge(
        removal_stats,
        left_on="nfl_id",
        right_on="defender_nfl_id",
        how="left"
    )

    df["intervention_removal"] = df["intervention_removal"].fillna(0).astype(int)
    df["impact_removal"] = df["impact_removal"].fillna(0)

    df.drop(columns=["defender_nfl_id"], inplace=True, errors="ignore")

    return df, df_detailed_results


df_base, df_detailed_results = build_base_dataframe(
    df, nfl_players, nfl_teams, df_detailed_results
)

# --------------------------------------------------
# SIDEBAR CONTROLS (STREAMLIT REPLACEMENT)
# --------------------------------------------------
with st.sidebar:
    st.header("Filters")

    position_value = st.selectbox(
        "Position",
        ["all", "LB", "CB", "S", "Other"]
    )

    plays_value = st.selectbox(
        "Play Volume",
        [0, 20, 50, 80],
        index=2
    )

    sort_value = st.selectbox(
        "Sort By",
        [
            "impact_removal",
            "avg_attention",
            "high_attention_pct",
            "play_count",
            "display_name"
        ]
    )

# --------------------------------------------------
# FILTER + SORT (UNCHANGED LOGIC)
# --------------------------------------------------
df_view = df_base.copy()

if position_value != "all":
    df_view = df_view[df_view["position_group"] == position_value]

if plays_value > 0 and len(df_view) > 0:
    threshold = (plays_value / 100) * df_view["play_count"].max()
    df_view = df_view[df_view["play_count"] >= threshold]

ascending = sort_value == "display_name"
df_view = df_view.sort_values(sort_value, ascending=ascending)
df_view["rank"] = range(1, len(df_view) + 1)

st.caption(f"Showing {len(df_view)} players")

# ===== CSS Styles =====
css_styles = """
    <style>
        .attention-table-container {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 100%);
            padding: 20px;
            border-radius: 12px;
            color: #fff;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-top: 20px;
        }
        
        .table-header {
            text-align: center;
            margin-bottom: 25px;
            padding: 20px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
            border-radius: 10px;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }
        
        .header-icon {
            font-size: 36px;
            margin-bottom: 10px;
        }
        
        .header-text h1 {
            margin: 0;
            font-size: 28px;
            font-weight: bold;
            color: #fff;
            line-height: 1.3;
            background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header-text p {
            margin: 8px 0 0 0;
            color: #9ca3af;
            font-size: 13px;
            font-weight: 500;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            background: rgba(30, 30, 50, 0.5);
            backdrop-filter: blur(10px);
            border: 1px solid #374151;
            border-radius: 8px;
            overflow: visible;
            margin-top: 20px;
        }
        
        thead {
            background: linear-gradient(90deg, rgba(55, 65, 81, 0.8) 0%, rgba(55, 65, 81, 0.4) 100%);
            border-bottom: 2px solid #4b5563;
        }
        
        th {
            padding: 12px 8px;
            text-align: center;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #9ca3af;
        }
        
        td {
            padding: 10px 8px;
            border-bottom: 1px solid #2d3748;
            vertical-align: middle;
            text-align: center;
        }
        
        tbody tr {
            transition: all 0.2s ease;
        }
        
        tbody tr:hover {
            background: rgba(59, 130, 246, 0.1);
        }
        
        tbody tr:last-child td {
            border-bottom: none;
        }
        
        .rank-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 6px;
            background: #374151;
            font-weight: bold;
            font-size: 14px;
            position: relative;
        }
        
        .rank-badge.gold {
            background: #fbbf24;
            color: #78350f;
        }
        
        .rank-badge.silver {
            background: #d1d5db;
            color: #374151;
        }
        
        .rank-badge.bronze {
            background: #d97706;
            color: #fff;
        }
        
        .medal {
            position: absolute;
            top: -6px;
            right: -6px;
            font-size: 14px;
        }
        
        .player-cell {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            text-align: center;
        }
        
        .player-headshot {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid;
            flex-shrink: 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            position: relative;
        }
        
        .team-dot {
            position: absolute;
            bottom: -2px;
            right: -2px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            border: 2px solid #1f2937;
        }
        
        .player-info h3 {
            margin: 0;
            font-size: 11px;
            font-weight: 600;
            color: #fff;
            white-space: nowrap;
        }
        
        .player-info p {
            margin: 2px 0 0 0;
            font-size: 9px;
            color: #9ca3af;
            white-space: nowrap;
        }
        
        .position-badge {
            display: inline-block;
            background: #374151;
            color: #e5e7eb;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 8px;
            font-weight: 600;
            margin-right: 3px;
        }
        
        .team-badge {
            display: inline-block;
            color: #fff;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 8px;
            font-weight: 600;
        }
        
        .metric-label {
            font-size: 8px;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.2px;
            margin-bottom: 2px;
        }
        
        .metric-value {
            font-size: 12px;
            font-weight: 700;
            color: #fff;
        }
        
        .attention-bar {
            width: 100%;
            height: 4px;
            background: #374151;
            border-radius: 2px;
            margin-top: 4px;
            overflow: hidden;
        }
        
        .attention-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s ease;
        }
        
        .intervention-header {
            cursor: help;
            position: relative;
        }
        
        .legend-container {
            margin-top: 20px;
            padding: 15px;
            background: rgba(30, 30, 50, 0.5);
            border: 1px solid #374151;
            border-radius: 8px;
        }
        
        .legend-title {
            font-size: 12px;
            font-weight: 600;
            color: #e5e7eb;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }
        
        .legend-item {
            margin-bottom: 8px;
            font-size: 10px;
            color: #d1d5db;
            line-height: 1.4;
        }
        
        .legend-item-title {
            font-weight: 600;
            color: #e5e7eb;
        }
        
        .legend-item-desc {
            color: #9ca3af;
            margin-top: 2px;
        }
        
        .no-results {
            padding: 40px;
            text-align: center;
            color: #9ca3af;
            font-size: 14px;
        }
    </style>
    """

# --------------------------------------------------
# ORIGINAL HTML TABLE FUNCTION (UNCHANGED)
# --------------------------------------------------
def create_table_html(filtered_df):

    if len(filtered_df) == 0:
        return f"""
        {css_styles}
        <div class="attention-table-container">
            <div class="no-results">
                No players match your current filters.
            </div>
        </div>
        """

    max_impact = df_detailed_results["impact_score"].max() / 0.07

    html = f"""
    {css_styles}
    <div class="attention-table-container">
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Player</th>
                    <th>High Attn %</th>
                    <th>Avg Attention</th>
                    <th>Impact</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, row in filtered_df.iterrows():
        avg_attention = row["avg_attention"] * 100
        high_attention = row["high_attention_pct"]
        avg_impact = row["impact_removal"] * 100
        impact_width = min((avg_impact / max_impact) * 100, 100)

        html += f"""
        <tr>
            <td>{row["rank"]}</td>
            <td>{row["display_name"]}</td>
            <td>{high_attention:.1f}%</td>
            <td>{avg_attention:.1f}%</td>
            <td>
                <div class="attention-bar">
                    <div class="attention-fill" style="width:{impact_width:.1f}%"></div>
                </div>
                {avg_impact:.1f}%
            </td>
        </tr>
        """

    html += """
            </tbody>
        </table>
    </div>
    """

    return html


# --------------------------------------------------
# RENDER (STREAMLIT)
# --------------------------------------------------
st.components.v1.html(
    create_table_html(df_view),
    height=1400,
    scrolling=True
)

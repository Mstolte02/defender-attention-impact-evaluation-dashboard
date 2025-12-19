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
df = pd.read_csv("defender_attention_all.csv")
nfl_players = nfl.import_players()
nfl_teams = nfl.import_team_desc()
df_detailed_results = pd.read_csv("intervention_detailed_results.csv")

# --------------------------------------------------
# BUILD BASE DATAFRAME
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
# SIDEBAR FILTERS
# --------------------------------------------------
with st.sidebar:
    st.header("Filters")
    position_value = st.selectbox(
        "Position", ["all", "LB", "CB", "S", "Other"]
    )
    plays_value = st.selectbox("Play Volume", [0, 20, 50, 80], index=2)
    sort_value = st.selectbox(
        "Sort By",
        ["impact_removal", "avg_attention", "high_attention_pct", "play_count", "display_name"]
    )

# --------------------------------------------------
# APPLY FILTERS + SORT
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

# --------------------------------------------------
# FULL CSS FOR STREAMLIT
# --------------------------------------------------
css_styles = """
<style>
.attention-table-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    padding: 20px;
    border-radius: 12px;
    color: #fff;
    background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 100%);
}
table {
    width: 100%;
    border-collapse: collapse;
    background: rgba(30,30,50,0.9);
    border-radius: 8px;
    overflow: hidden;
}
thead {
    background: linear-gradient(90deg, rgba(55,65,81,0.8) 0%, rgba(55,65,81,0.4) 100%);
}
th, td {
    padding: 8px 6px;
    text-align: center;
    font-size: 12px;
    color: #fff;
}
tbody tr:hover {
    background: rgba(59,130,246,0.1);
}
.player-headshot {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 2px solid;
}
.rank-badge {
    display: inline-block;
    width: 28px;
    height: 28px;
    border-radius: 5px;
    background: #374151;
    font-weight: bold;
}
.rank-badge.gold {background: #fbbf24; color:#78350f;}
.rank-badge.silver {background:#d1d5db;color:#374151;}
.rank-badge.bronze {background:#d97706;color:#fff;}
.attention-bar {
    width: 100%;
    height: 6px;
    background: #374151;
    border-radius: 3px;
    margin-top: 4px;
}
.attention-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
</style>
"""

# --------------------------------------------------
# HTML TABLE FUNCTION
# --------------------------------------------------
def create_table_html(df):
    if len(df) == 0:
        return f"{css_styles}<div class='attention-table-container'><div>No players match filters.</div></div>"

    max_impact = df_detailed_results["impact_score"].max() / 0.07
    html = f"{css_styles}<div class='attention-table-container'><table><thead><tr><th>Rank</th><th>Player</th><th>High Attn %</th><th>Avg Attention</th><th>Impact</th></tr></thead><tbody>"

    for _, row in df.iterrows():
        avg_attention = row["avg_attention"] * 100
        high_attention = row["high_attention_pct"]
        avg_impact = row["impact_removal"] * 100
        impact_width = min((avg_impact / max_impact) * 100, 100)

        html += f"""
        <tr>
            <td><div class='rank-badge'>{row['rank']}</div></td>
            <td>{row['display_name']}</td>
            <td>{high_attention:.1f}%</td>
            <td>{avg_attention:.1f}%</td>
            <td>
                <div class='attention-bar'>
                    <div class='attention-fill' style='width:{impact_width:.1f}%;'></div>
                </div>
                {avg_impact:.1f}%
            </td>
        </tr>
        """
    html += "</tbody></table></div>"
    return html

# --------------------------------------------------
# RENDER STREAMLIT COMPONENT
# --------------------------------------------------
st.components.v1.html(create_table_html(df_view), height=1400, scrolling=True)

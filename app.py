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
df = pd.read_csv("/Users/markstolte/Downloads/Big Data Bowl Submission/defender_attention_all.csv")
nfl_players = nfl.import_players()
nfl_teams = nfl.import_team_desc()
df_detailed_results = pd.read_csv("/Users/markstolte/Downloads/Big Data Bowl Submission/intervention_detailed_results.csv")

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

# --------------------------------------------------
# ORIGINAL CSS (UNCHANGED)
# --------------------------------------------------
css_styles = """<style>
<YOUR FULL ORIGINAL CSS HERE — UNCHANGED>
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

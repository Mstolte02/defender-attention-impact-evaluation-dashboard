import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# Version 2.0 - Updated styling with Oswald font and larger elements

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
        df_detailed_results = df_detailed_results[df_detailed_results['intervention_type'] == 'removal']
        df_detailed_results['defender_nfl_id'] = df_detailed_results['defender_nfl_id'].astype(str)
        
        # Count interventions by type for each player and calculate average impact score
        removal_stats = df_detailed_results.groupby('defender_nfl_id').agg(
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
        total_interventions = df_detailed_results.groupby('defender_nfl_id').size().reset_index(name='total_interventions')
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
    
    # ===== CSS Styles =====
    css_styles = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&display=swap');
        
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
            margin-bottom: 30px;
            padding: 25px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
            border-radius: 10px;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }
        
        .header-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        
        .header-text h1 {
            margin: 0 0 20px 0;
            font-family: 'Oswald', sans-serif;
            font-size: 36px;
            font-weight: 600;
            color: #fff;
            line-height: 1.3;
            text-align: center;
            background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .metric-description {
            text-align: left;
            color: #d1d5db;
            font-size: 14px;
            line-height: 1.8;
            max-width: 900px;
            margin: 0 auto;
            padding: 15px 20px;
            background: rgba(30, 41, 59, 0.4);
            border-radius: 8px;
            border-left: 3px solid #3b82f6;
        }
        
        .metric-description strong {
            color: #60a5fa;
            font-weight: 600;
        }
        
        .metric-description p {
            margin: 8px 0;
        }
        
        .header-text p {
            margin: 8px 0 0 0;
            color: #9ca3af;
            font-size: 15px;
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
            padding: 18px 12px;
            text-align: center;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #9ca3af;
        }
        
        td {
            padding: 16px 12px;
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
            width: 42px;
            height: 42px;
            border-radius: 8px;
            background: #374151;
            font-weight: bold;
            font-size: 18px;
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
            top: -8px;
            right: -8px;
            font-size: 18px;
        }
        
        .player-cell {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            text-align: center;
        }
        
        .player-headshot {
            width: 48px;
            height: 48px;
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
            width: 14px;
            height: 14px;
            border-radius: 50%;
            border: 2px solid #1f2937;
        }
        
        .player-info h3 {
            margin: 0;
            font-size: 15px;
            font-weight: 600;
            color: #fff;
            white-space: nowrap;
        }
        
        .player-info p {
            margin: 4px 0 0 0;
            font-size: 12px;
            color: #9ca3af;
            white-space: nowrap;
        }
        
        .position-badge {
            display: inline-block;
            background: #374151;
            color: #e5e7eb;
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-right: 4px;
        }
        
        .team-badge {
            display: inline-block;
            color: #fff;
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        
        .metric-label {
            font-size: 11px;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            margin-bottom: 4px;
        }
        
        .metric-value {
            font-size: 16px;
            font-weight: 700;
            color: #fff;
        }
        
        .attention-bar {
            width: 100%;
            height: 6px;
            background: #374151;
            border-radius: 3px;
            margin-top: 6px;
            overflow: hidden;
        }
        
        .attention-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        
        .no-results {
            padding: 40px;
            text-align: center;
            color: #9ca3af;
            font-size: 14px;
        }
        
        /* Player detail modal/tooltip */
        .player-detail-popup {
            display: none;
            position: absolute;
            background: rgba(15, 23, 42, 0.98);
            border: 2px solid #3b82f6;
            border-radius: 12px;
            padding: 20px;
            z-index: 1000;
            max-width: 350px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
        }
        
        .player-detail-popup.active {
            display: block;
        }
        
        .popup-header {
            font-size: 18px;
            font-weight: 700;
            color: #fff;
            margin-bottom: 15px;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 10px;
        }
        
        .popup-category {
            margin: 12px 0;
            padding: 12px;
            background: rgba(59, 130, 246, 0.1);
            border-radius: 8px;
            border-left: 3px solid;
        }
        
        .popup-category-title {
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        
        .popup-category-desc {
            font-size: 12px;
            color: #d1d5db;
            line-height: 1.5;
        }
        
        .player-name-clickable {
            cursor: pointer;
            transition: color 0.2s ease;
        }
        
        .player-name-clickable:hover {
            color: #60a5fa;
            text-decoration: underline;
        }
    </style>
    """
    
    # ===== Streamlit UI Components =====
    
    # Create search and filter row
    col1, col2 = st.columns(2)
    
    with col1:
        search_query = st.text_input(
            '🔍 Search Player',
            placeholder='Type player name...',
            key='player_search'
        )
    
    with col2:
        # Get unique teams from the data
        all_teams = sorted(df_with_players['latest_team'].dropna().unique().tolist())
        team_filter = st.selectbox(
            'Team',
            options=['All Teams'] + all_teams,
            index=0
        )
    
    # Create three columns for other filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        position_filter = st.selectbox(
            'Position',
            options=['All Positions', 'LB (Linebackers)', 'CB (Cornerbacks)', 'S (Safeties)', 'Other Positions'],
            index=0
        )
    
    with col2:
        max_plays = df_with_players['play_count'].max() if len(df_with_players) > 0 else 1
        plays_filter = st.selectbox(
            'Play Volume',
            options=['No Filter', '≥ 20% of Max Plays', '≥ 50% of Max Plays', '≥ 80% of Max Plays'],
            index=2
        )
    
    with col3:
        sort_option = st.selectbox(
            'Sort by',
            options=[
                'Impact (High to Low)',
                'Avg Attention (High to Low)',
                'High Attention % (High to Low)',
                'Play Count (High to Low)',
                'Player Name (A-Z)'
            ],
            index=0
        )
    
    # Map UI selections to data operations
    position_map = {
        'All Positions': 'all',
        'LB (Linebackers)': 'LB',
        'CB (Cornerbacks)': 'CB',
        'S (Safeties)': 'S',
        'Other Positions': 'Other'
    }
    
    plays_map = {
        'No Filter': 0,
        '≥ 20% of Max Plays': 20,
        '≥ 50% of Max Plays': 50,
        '≥ 80% of Max Plays': 80
    }
    
    sort_map = {
        'Avg Attention (High to Low)': 'avg_attention',
        'High Attention % (High to Low)': 'high_attention_pct',
        'Play Count (High to Low)': 'play_count',
        'Impact (High to Low)': 'impact_removal',
        'Player Name (A-Z)': 'display_name'
    }
    
    # Apply filters
    filtered_df = df_with_players.copy()
    
    # Search filter
    if search_query:
        filtered_df = filtered_df[
            filtered_df['display_name'].str.contains(search_query, case=False, na=False)
        ]
    
    # Team filter
    if team_filter != 'All Teams':
        filtered_df = filtered_df[filtered_df['latest_team'] == team_filter]
    
    # Position filter
    selected_position = position_map[position_filter]
    if selected_position != 'all':
        filtered_df = filtered_df[filtered_df['position_group'] == selected_position]
    
    # Plays filter
    plays_threshold = plays_map[plays_filter]
    if plays_threshold > 0:
        current_max_plays = filtered_df['play_count'].max() if len(filtered_df) > 0 else 1
        threshold = (plays_threshold / 100) * current_max_plays
        filtered_df = filtered_df[filtered_df['play_count'] >= threshold]
    
    # Apply sorting
    sort_column = sort_map[sort_option]
    ascending = sort_column == 'display_name'
    
    if sort_column in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_column, ascending=ascending)
    
    # Add rank based on sort
    filtered_df = filtered_df.copy()
    filtered_df['rank'] = range(1, len(filtered_df) + 1)
    
    # ===== PAGINATION =====
    # Set items per page
    items_per_page = 50
    total_items = len(filtered_df)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
    # Page selector
    if total_items > items_per_page:
        page_number = st.selectbox(
            f'Page (showing {items_per_page} players per page)',
            options=list(range(1, total_pages + 1)),
            index=0,
            key='page_selector'
        )
    else:
        page_number = 1
    
    # Calculate pagination indices
    start_idx = (page_number - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    # Get paginated data
    paginated_df = filtered_df.iloc[start_idx:end_idx].copy()
    
    # Display results count with pagination info
    if total_items > items_per_page:
        st.info(f"Showing {start_idx + 1}-{end_idx} of {total_items} players (Page {page_number} of {total_pages}) | Sorted by {sort_option}")
    else:
        st.info(f"Showing {total_items} of {len(df_with_players)} players | Sorted by {sort_option}")
    
    # ===== Create HTML Table =====
    def get_player_category(avg_attention, impact_removal, df_data):
        """Determine player category based on attention and impact percentiles"""
        # Calculate percentiles from the full dataset
        attention_median = df_data['avg_attention'].median()
        impact_median = df_data['impact_removal'].median()
        
        high_attention = avg_attention >= attention_median
        high_impact = impact_removal >= impact_median
        
        if high_attention and high_impact:
            return {
                'title': 'High Attention · High Impact',
                'desc': 'Consistently involved and directly influenced the pass outcome.',
                'color': '#10b981'
            }
        elif not high_attention and high_impact:
            return {
                'title': 'Low Attention · High Impact',
                'desc': 'Not frequently involved, but made a decisive impact when it mattered.',
                'color': '#f59e0b'
            }
        elif high_attention and not high_impact:
            return {
                'title': 'High Attention · Low Impact',
                'desc': 'Often involved, but did not materially affect the result.',
                'color': '#6366f1'
            }
        else:
            return {
                'title': 'Low Attention · Low Impact',
                'desc': 'Rarely factored into the play and had minimal effect on the outcome.',
                'color': '#6b7280'
            }
    
    def create_table_html(filtered_df, full_df):
        """Create HTML table for the filtered dataframe with bars for all metrics"""
        if len(filtered_df) == 0:
            return f"""
            {css_styles}
            <div class="attention-table-container">
                <div class="no-results">
                    No players match your current filters. Try adjusting the filter settings.
                </div>
            </div>
            """

        html_content = f"""
        {css_styles}
        <div class="attention-table-container">
            <div class="table-header">
                <div class="header-icon">🏈</div>
                <div class="header-text">
                    <h1>Neural Network Attention & Impact Analysis:<br>Top Defenders in Pass Coverage in 2023</h1>
                    <div class="metric-description">
                        <p><strong>Attention:</strong> Mean attention weight assigned to the defender across all post-throw frames. High value → defender had potential to make a play in critical moments</p>
                        <p><strong>High Attention %:</strong> Percentage of post-throw frames where the defender was among the model's most influential defenders. High value → defender frequently involved in the play outcome</p>
                        <p><strong>Removal Impact:</strong> Change in predicted completion probability when the defender is removed from the play. Measures most valuable defenders in coverage</p>
                    </div>
                </div>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th style="width: 5%;">Rank</th>
                        <th style="width: 40%;">Player</th>
                        <th style="width: 13%;">High Attn %</th>
                        <th style="width: 13%;">Avg Attention</th>
                        <th style="width: 29%;">REMOVAL<br><span style="font-size: 11px; text-transform: none; color: #6b7280;">Impact</span></th>
                    </tr>
                </thead>
                <tbody>
        """

        max_impact = df_detailed_results['impact_score'].max() / 0.07 if df_detailed_results is not None else 1

        for idx, row in filtered_df.iterrows():
            rank = int(row['rank'])
            
            # Medal
            medal_class = ''
            medal_emoji = ''
            if rank == 1: medal_class, medal_emoji = 'gold', '🥇'
            elif rank == 2: medal_class, medal_emoji = 'silver', '🥈'
            elif rank == 3: medal_class, medal_emoji = 'bronze', '🥉'

            # Player info
            player_name = str(row.get('display_name', 'Unknown')).replace('<','&lt;').replace('>','&gt;')
            position = str(row.get('position', 'N/A')).replace('<','&lt;').replace('>','&gt;')
            team = str(row.get('latest_team', 'N/A')).replace('<','&lt;').replace('>','&gt;')
            headshot = row.get('headshot', '')
            team_color = row.get('team_color', '#374151')
            team_color2 = row.get('team_color2', '#6b7280')

            # Metrics
            avg_attention = float(row.get('avg_attention', 0)) * 100
            high_attention_pct = float(row.get('high_attention_pct', 0))
            avg_impact = float(row.get('impact_removal', 0)) * 100
            play_count = int(row.get('play_count', 0))

            # Bar widths
            high_attention_width = min(high_attention_pct, 100)
            avg_attention_width = min(avg_attention * 100, 100)
            impact_width = min((avg_impact / max_impact) * 100, 100) if max_impact > 0 else 0

            # Headshot HTML
            if pd.notna(headshot) and headshot and isinstance(headshot, str):
                headshot_html = f'<img src="{headshot}" alt="{player_name}" class="player-headshot" style="border-color: {team_color};" onerror="this.style.display=\'none\';">'
            else:
                initials = ''.join([w[0].upper() for w in player_name.split() if w])[:2] or '??'
                headshot_html = f'<div class="player-headshot" style="border-color: {team_color}; background: linear-gradient(135deg, {team_color}, {team_color2}); display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 16px; color: #fff;">{initials}</div>'

            # Get player category
            category = get_player_category(row.get('avg_attention', 0), row.get('impact_removal', 0), full_df)
            
            # Create unique ID for popup
            popup_id = f"popup-{idx}"
            
            html_content += f"""
                <tr>
                    <td>
                        <div class="rank-badge {medal_class}">
                            {rank}{f'<span class="medal">{medal_emoji}</span>' if medal_emoji else ''}
                        </div>
                    </td>
                    <td style="position: relative;">
                        <div class="player-cell">
                            <div style="position: relative;">
                                {headshot_html}
                                <div class="team-dot" style="background-color: {team_color2};"></div>
                            </div>
                            <div class="player-info">
                                <h3 class="player-name-clickable" onclick="togglePopup('{popup_id}')" title="Click for player insights">{player_name}</h3>
                                <p>
                                    <span class="position-badge">{position}</span>
                                    <span class="team-badge" style="background-color: {team_color};">{team}</span>
                                    <span style="color: #6b7280; font-size: 11px;">{play_count} plays</span>
                                </p>
                            </div>
                        </div>
                        
                        <!-- Player detail popup -->
                        <div id="{popup_id}" class="player-detail-popup">
                            <div class="popup-header">{player_name}</div>
                            <div class="popup-category" style="border-left-color: {category['color']};">
                                <div class="popup-category-title" style="color: {category['color']};">{category['title']}</div>
                                <div class="popup-category-desc">{category['desc']}</div>
                            </div>
                        </div>
                    </td>

                    <!-- High Attention % with bar -->
                    <td>
                        <div class="metric-label">High Attention</div>
                        <div class="metric-value">{high_attention_pct:.1f}%</div>
                        <div class="attention-bar">
                            <div class="attention-fill" style="width: {high_attention_width:.1f}%; background: linear-gradient(90deg, {team_color}, {team_color2});"></div>
                        </div>
                    </td>

                    <!-- Avg Attention with bar -->
                    <td>
                        <div class="metric-label">Avg Attention</div>
                        <div class="metric-value">{avg_attention:.1f}%</div>
                        <div class="attention-bar">
                            <div class="attention-fill" style="width: {avg_attention_width:.1f}%; background: linear-gradient(90deg, {team_color}, {team_color2});"></div>
                        </div>
                    </td>

                    <!-- Impact Removal - value only, no bar -->
                    <td>
                        <div class="metric-label">Impact</div>
                        <div class="metric-value">{avg_impact:.1f}%</div>
                    </td>
                </tr>
            """

        html_content += """
                </tbody>
            </table>
        </div>
        
        <script>
        function togglePopup(popupId) {
            // Close all other popups first
            const allPopups = document.querySelectorAll('.player-detail-popup');
            allPopups.forEach(popup => {
                if (popup.id !== popupId) {
                    popup.classList.remove('active');
                }
            });
            
            // Toggle the clicked popup
            const popup = document.getElementById(popupId);
            if (popup) {
                popup.classList.toggle('active');
            }
        }
        
        // Close popups when clicking outside
        document.addEventListener('click', function(event) {
            if (!event.target.closest('.player-name-clickable') && !event.target.closest('.player-detail-popup')) {
                const allPopups = document.querySelectorAll('.player-detail-popup');
                allPopups.forEach(popup => {
                    popup.classList.remove('active');
                });
            }
        });
        </script>
        """
        return html_content
    
    # Display the table using components.html for proper rendering
    html_content = create_table_html(paginated_df, df_with_players)
    
    # Calculate height based on number of rows (roughly 80px per row + 500px for header/footer)
    table_height = min(3000, max(600, len(paginated_df) * 80 + 500))
    
    # Add cache-busting parameter to force reload
    import random
    cache_bust = random.randint(1, 1000000)
    html_with_cache_bust = f"""
    <!-- Cache bust: {cache_bust} -->
    {html_content}
    """
    
    components.html(html_with_cache_bust, height=table_height, scrolling=True)


# ===== USAGE EXAMPLE FOR STREAMLIT APP =====
# Create a file named app.py with the following:
"""
import streamlit as st
import pandas as pd
from player_attention_table_streamlit import create_player_attention_table

# Set page config
st.set_page_config(
    page_title="NFL Player Attention Analysis",
    page_icon="🏈",
    layout="wide"
)

# Load your data
# df = pd.read_csv('player_attention_metrics.csv')
# nfl_players = pd.read_csv('nfl_players.csv')
# nfl_teams = pd.read_csv('nfl_teams.csv')
# df_detailed_results = pd.read_csv('detailed_results.csv')

# Create the table
create_player_attention_table(df, nfl_players, nfl_teams, df_detailed_results)
"""

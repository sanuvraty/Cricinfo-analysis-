"""
Cricket Statistics Calculations Module

This module builds on the data processing module to implement specialized cricket
statistics calculations for player comparisons and analysis, particularly focused
on comparing players like Sachin Tendulkar and Virat Kohli.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from statsguru_processor import StatguruDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CricketStatsCalculator:
    """
    Implements specialized cricket statistics calculations for player comparisons and analysis.
    """
    
    def __init__(self):
        """Initialize the cricket statistics calculator."""
        self.processor = StatguruDataProcessor()
        self.output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def compare_batting_careers(self, player1_df: pd.DataFrame, player2_df: pd.DataFrame, 
                               player1_name: str, player2_name: str) -> Dict[str, Any]:
        """
        Comprehensive comparison of two batsmen's careers
        
        Args:
            player1_df: DataFrame containing first player's statistics
            player2_df: DataFrame containing second player's statistics
            player1_name: Name of first player
            player2_name: Name of second player
            
        Returns:
            dict: Dictionary with comprehensive comparison results
        """
        try:
            if player1_df is None or player2_df is None:
                logger.error("Player dataframes cannot be None")
                return None
            
            # Basic metrics comparison
            basic_metrics = ['Runs', 'Average', 'Strike_Rate', 'Hundreds', 'Fifties', 'Innings', 'Matches']
            basic_comparison = self.processor.compare_players(player1_df, player2_df, basic_metrics)
            
            # Calculate impact scores
            player1_impact = self.processor.calculate_batting_impact_score(player1_df)
            player2_impact = self.processor.calculate_batting_impact_score(player2_df)
            
            # Calculate batting indices
            player1_index = self.processor.calculate_batting_index(player1_df)
            player2_index = self.processor.calculate_batting_index(player2_df)
            
            # Performance by opposition
            if 'Opposition' in player1_df.columns and 'Opposition' in player2_df.columns:
                player1_by_opposition = self.processor.calculate_performance_by_condition(
                    player1_df, 'Opposition', 'Average'
                )
                player2_by_opposition = self.processor.calculate_performance_by_condition(
                    player2_df, 'Opposition', 'Average'
                )
            else:
                player1_by_opposition = None
                player2_by_opposition = None
            
            # Performance by ground/venue
            if 'Ground' in player1_df.columns and 'Ground' in player2_df.columns:
                player1_by_ground = self.processor.calculate_performance_by_condition(
                    player1_df, 'Ground', 'Average'
                )
                player2_by_ground = self.processor.calculate_performance_by_condition(
                    player2_df, 'Ground', 'Average'
                )
            else:
                player1_by_ground = None
                player2_by_ground = None
            
            # Career progression
            if 'Date' in player1_df.columns and 'Date' in player2_df.columns:
                player1_progression = self.processor.calculate_career_progression(
                    player1_df, 'Average', 'year'
                )
                player2_progression = self.processor.calculate_career_progression(
                    player2_df, 'Average', 'year'
                )
            else:
                player1_progression = None
                player2_progression = None
            
            # Compile comprehensive comparison
            comparison = {
                'basic_metrics': basic_comparison,
                'impact_scores': {
                    player1_name: player1_impact,
                    player2_name: player2_impact,
                    'difference': player1_impact - player2_impact if player1_impact and player2_impact else None
                },
                'batting_indices': {
                    player1_name: player1_index,
                    player2_name: player2_index,
                    'difference': player1_index - player2_index if player1_index and player2_index else None
                },
                'by_opposition': {
                    player1_name: player1_by_opposition,
                    player2_name: player2_by_opposition
                },
                'by_ground': {
                    player1_name: player1_by_ground,
                    player2_name: player2_by_ground
                },
                'career_progression': {
                    player1_name: player1_progression,
                    player2_name: player2_progression
                }
            }
            
            return comparison
        except Exception as e:
            logger.error(f"Error comparing batting careers: {e}")
            return None
    
    def calculate_era_adjusted_stats(self, player_df: pd.DataFrame, era_df: pd.DataFrame, 
                                   metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate era-adjusted statistics
        
        Args:
            player_df: DataFrame containing player's statistics
            era_df: DataFrame containing era statistics
            metrics: List of metrics to adjust
            
        Returns:
            dict: Dictionary with era-adjusted statistics
        """
        try:
            if player_df is None or era_df is None:
                logger.error("Player or era dataframe cannot be None")
                return None
            
            # Default metrics if none provided
            if metrics is None:
                # Try to determine if batting or bowling data
                if 'Wickets' in player_df.columns:
                    # Bowling data
                    metrics = ['Average', 'Economy', 'Strike_Rate']
                else:
                    # Batting data
                    metrics = ['Average', 'Strike_Rate']
            
            era_adjusted = {}
            
            for metric in metrics:
                if metric in player_df.columns and metric in era_df.columns:
                    benchmark = self.processor.calculate_era_benchmark(player_df, era_df, metric)
                    
                    if benchmark:
                        # For batting metrics, higher is better
                        batting_metrics = ['Runs', 'Average', 'Strike_Rate', 'Hundreds', 'Fifties']
                        
                        if metric in batting_metrics:
                            # Adjust upward if player is below era average
                            if benchmark['relative_performance'] < 1:
                                adjusted_value = benchmark['player_metric'] * (1 / benchmark['relative_performance'])
                            else:
                                adjusted_value = benchmark['player_metric']
                        else:
                            # For bowling metrics, lower is better
                            # Adjust downward if player is above era average
                            if benchmark['relative_performance'] < 1:
                                adjusted_value = benchmark['player_metric'] * benchmark['relative_performance']
                            else:
                                adjusted_value = benchmark['player_metric']
                        
                        era_adjusted[metric] = {
                            'raw_value': benchmark['player_metric'],
                            'era_average': benchmark['era_metric'],
                            'relative_performance': benchmark['relative_performance'],
                            'adjusted_value': round(adjusted_value, 2)
                        }
                else:
                    logger.warning(f"Metric {metric} not found in both dataframes")
            
            return era_adjusted
        except Exception as e:
            logger.error(f"Error calculating era-adjusted stats: {e}")
            return None
    
    def calculate_weighted_batting_rating(self, df: pd.DataFrame, weights: Dict[str, float] = None) -> float:
        """
        Calculate weighted batting rating
        
        Args:
            df: DataFrame containing batting statistics
            weights: Dictionary mapping metrics to weights
            
        Returns:
            float: Calculated weighted batting rating
        """
        try:
            # Default weights if none provided
            if weights is None:
                weights = {
                    'Average': 0.35,
                    'Strike_Rate': 0.25,
                    'Runs': 0.20,
                    'Hundreds': 0.15,
                    'Fifties': 0.05
                }
            
            # Check for required columns
            missing_cols = [col for col in weights.keys() if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns for weighted rating: {missing_cols}")
                # Adjust weights to exclude missing columns
                total_weight = sum([w for c, w in weights.items() if c not in missing_cols])
                weights = {c: w/total_weight for c, w in weights.items() if c not in missing_cols}
            
            # Normalize metrics to 0-100 scale
            normalized_df = df.copy()
            
            # Normalization factors (based on world-class benchmarks)
            norm_factors = {
                'Average': 60,  # World-class average
                'Strike_Rate': 100,  # World-class strike rate
                'Runs': 10000,  # World-class career runs
                'Hundreds': 30,  # World-class career hundreds
                'Fifties': 50   # World-class career fifties
            }
            
            for metric in weights.keys():
                if metric in df.columns:
                    if metric in ['Runs', 'Hundreds', 'Fifties']:
                        # Sum metrics
                        value = df[metric].sum()
                    else:
                        # Average metrics
                        value = df[metric].mean()
                    
                    # Normalize to 0-100 scale
                    normalized_value = min(value / norm_factors.get(metric, 1), 1) * 100
                    normalized_df[f'norm_{metric}'] = normalized_value
            
            # Calculate weighted rating
            rating = 0
            for metric, weight in weights.items():
                if f'norm_{metric}' in normalized_df.columns:
                    rating += normalized_df[f'norm_{metric}'].iloc[0] * weight
            
            return round(rating, 2)
        except Exception as e:
            logger.error(f"Error calculating weighted batting rating: {e}")
            return None
    
    def calculate_match_winning_contribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate match-winning contributions
        
        Args:
            df: DataFrame containing player statistics with match result information
            
        Returns:
            dict: Dictionary with match-winning contribution analysis
        """
        try:
            if 'Result' not in df.columns:
                logger.error("Match result information not available")
                return None
            
            # Filter for wins, losses, and draws
            wins_df = df[df['Result'].str.contains('won', case=False, na=False)]
            losses_df = df[df['Result'].str.contains('lost', case=False, na=False)]
            draws_df = df[df['Result'].str.contains('draw', case=False, na=False)]
            
            # Calculate metrics for each result type
            metrics = {}
            
            # Batting metrics
            if 'Runs' in df.columns and 'Average' in df.columns:
                metrics['batting'] = {
                    'wins': {
                        'matches': len(wins_df),
                        'runs': wins_df['Runs'].sum(),
                        'average': wins_df['Average'].mean() if not wins_df.empty else 0,
                        'hundreds': wins_df['Hundreds'].sum() if 'Hundreds' in wins_df.columns else 0
                    },
                    'losses': {
                        'matches': len(losses_df),
                        'runs': losses_df['Runs'].sum(),
                        'average': losses_df['Average'].mean() if not losses_df.empty else 0,
                        'hundreds': losses_df['Hundreds'].sum() if 'Hundreds' in losses_df.columns else 0
                    },
                    'draws': {
                        'matches': len(draws_df),
                        'runs': draws_df['Runs'].sum(),
                        'average': draws_df['Average'].mean() if not draws_df.empty else 0,
                        'hundreds': draws_df['Hundreds'].sum() if 'Hundreds' in draws_df.columns else 0
                    }
                }
                
                # Calculate win contribution index
                if not wins_df.empty and not losses_df.empty:
                    win_avg = metrics['batting']['wins']['average']
                    loss_avg = metrics['batting']['losses']['average']
                    
                    if loss_avg > 0:
                        win_contribution_index = win_avg / loss_avg
                    else:
                        win_contribution_index = win_avg if win_avg > 0 else 0
                    
                    metrics['batting']['win_contribution_index'] = round(win_contribution_index, 2)
            
            # Bowling metrics
            if 'Wickets' in df.columns and 'Average' in df.columns:
                metrics['bowling'] = {
                    'wins': {
                        'matches': len(wins_df),
                        'wickets': wins_df['Wickets'].sum(),
                        'average': wins_df['Average'].mean() if not wins_df.empty else 0,
                        'five_wickets': wins_df['FiveWickets'].sum() if 'FiveWickets' in wins_df.columns else 0
                    },
                    'losses': {
                        'matches': len(losses_df),
                        'wickets': losses_df['Wickets'].sum(),
                        'average': losses_df['Average'].mean() if not losses_df.empty else 0,
                        'five_wickets': losses_df['FiveWickets'].sum() if 'FiveWickets' in losses_df.columns else 0
                    },
                    'draws': {
                        'matches': len(draws_df),
                        'wickets': draws_df['Wickets'].sum(),
                        'average': draws_df['Average'].mean() if not draws_df.empty else 0,
                        'five_wickets': draws_df['FiveWickets'].sum() if 'FiveWickets' in draws_df.columns else 0
                    }
                }
                
                # Calculate win contribution index (for bowling, lower average is better)
                if not wins_df.empty and not losses_df.empty:
                    win_avg = metrics['bowling']['wins']['average']
                    loss_avg = metrics['bowling']['losses']['average']
                    
                    if win_avg > 0:
                        win_contribution_index = loss_avg / win_avg
                    else:
                        win_contribution_index = loss_avg if loss_avg > 0 else 0
                    
                    metrics['bowling']['win_contribution_index'] = round(win_contribution_index, 2)
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating match-winning contribution: {e}")
            return None
    
    def visualize_player_comparison(self, comparison: Dict[str, Any], player1_name: str, player2_name: str, 
                                  output_dir: str = None) -> Dict[str, str]:
        """
        Create visualizations for player comparison
        
        Args:
            comparison: Dictionary with comparison results
            player1_name: Name of first player
            player2_name: Name of second player
            output_dir: Directory to save visualizations
            
        Returns:
            dict: Dictionary mapping visualization types to file paths
        """
        try:
            if comparison is None:
                logger.error("Comparison data cannot be None")
                return None
            
            if output_dir is None:
                output_dir = self.output_dir
            
            os.makedirs(output_dir, exist_ok=True)
            
            visualization_paths = {}
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # 1. Basic metrics comparison
            if 'basic_metrics' in comparison and comparison['basic_metrics']:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                metrics = list(comparison['basic_metrics'].keys())
                player1_values = [comparison['basic_metrics'][m]['player1'] for m in metrics]
                player2_values = [comparison['basic_metrics'][m]['player2'] for m in metrics]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                ax.bar(x - width/2, player1_values, width, label=player1_name)
                ax.bar(x + width/2, player2_values, width, label=player2_name)
                
                ax.set_xticks(x)
                ax.set_xticklabels(metrics, rotation=45)
                ax.legend()
                
                ax.set_title(f'Basic Metrics Comparison: {player1_name} vs {player2_name}')
                ax.set_ylabel('Value')
                
                plt.tight_layout()
                
                # Save figure
                basic_metrics_path = os.path.join(output_dir, f'basic_metrics_comparison.png')
                plt.savefig(basic_metrics_path)
                plt.close(fig)
                
                visualization_paths['basic_metrics'] = basic_metrics_path
            
            # 2. Impact scores and batting indices
            if 'impact_scores' in comparison and 'batting_indices' in comparison:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Impact scores
                impact_scores = [
                    comparison['impact_scores'].get(player1_name, 0),
                    comparison['impact_scores'].get(player2_name, 0)
                ]
                
                ax1.bar([player1_name, player2_name], impact_scores, color=['#1f77b4', '#ff7f0e'])
                ax1.set_title('Batting Impact Scores')
                ax1.set_ylabel('Impact Score')
                
                # Batting indices
                batting_indices = [
                    comparison['batting_indices'].get(player1_name, 0),
                    comparison['batting_indices'].get(player2_name, 0)
                ]
                
                ax2.bar([player1_name, player2_name], batting_indices, color=['#1f77b4', '#ff7f0e'])
                ax2.set_title('Batting Indices')
                ax2.set_ylabel('Batting Index')
                
                plt.tight_layout()
                
                # Save figure
                impact_path = os.path.join(output_dir, f'impact_and_indices.png')
                plt.savefig(impact_path)
                plt.close(fig)
                
                visualization_paths['impact_and_indices'] = impact_path
            
            # 3. Career progression
            if 'career_progression' in comparison:
                player1_prog = comparison['career_progression'].get(player1_name)
                player2_prog = comparison['career_progression'].get(player2_name)
                
                if player1_prog is not None and player2_prog is not None:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot progression
                    ax.plot(player1_prog.iloc[:, 0], player1_prog['Average'], 
                           marker='o', linestyle='-', label=player1_name)
                    ax.plot(player2_prog.iloc[:, 0], player2_prog['Average'], 
                           marker='s', linestyle='-', label=player2_name)
                    
                    ax.set_title(f'Career Progression: {player1_name} vs {player2_name}')
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Batting Average')
                    ax.legend()
                    
                    plt.tight_layout()
                    
                    # Save figure
                    progression_path = os.path.join(output_dir, f'career_progression.png')
                    plt.savefig(progression_path)
                    plt.close(fig)
                    
                    visualization_paths['career_progression'] = progression_path
            
            # 4. Performance by opposition
            if 'by_opposition' in comparison:
                player1_opp = comparison['by_opposition'].get(player1_name)
                player2_opp = comparison['by_opposition'].get(player2_name)
                
                if player1_opp is not None and player2_opp is not None:
                    # Merge the dataframes
                    merged_df = pd.merge(
                        player1_opp, player2_opp, 
                        on='Opposition', 
                        how='outer', 
                        suffixes=(f'_{player1_name}', f'_{player2_name}')
                    )
                    
                    # Fill NaN values with 0
                    merged_df = merged_df.fillna(0)
                    
                    # Sort by average of player1
                    avg_col = f'Average_{player1_name}'
                    if avg_col in merged_df.columns:
                        merged_df = merged_df.sort_values(by=avg_col, ascending=False)
                    
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    x = np.arange(len(merged_df))
                    width = 0.35
                    
                    ax.bar(x - width/2, merged_df[f'Average_{player1_name}'], 
                          width, label=player1_name)
                    ax.bar(x + width/2, merged_df[f'Average_{player2_name}'], 
                          width, label=player2_name)
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(merged_df['Opposition'], rotation=45, ha='right')
                    ax.legend()
                    
                    ax.set_title(f'Performance by Opposition: {player1_name} vs {player2_name}')
                    ax.set_ylabel('Batting Average')
                    
                    plt.tight_layout()
                    
                    # Save figure
                    opposition_path = os.path.join(output_dir, f'performance_by_opposition.png')
                    plt.savefig(opposition_path)
                    plt.close(fig)
                    
                    visualization_paths['by_opposition'] = opposition_path
            
            return visualization_paths
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return None
    
    def generate_comparison_report(self, comparison: Dict[str, Any], player1_name: str, player2_name: str, 
                                 format_type: str, visualization_paths: Dict[str, str] = None) -> str:
        """
        Generate a comprehensive comparison report
        
        Args:
            comparison: Dictionary with comparison results
            player1_name: Name of first player
            player2_name: Name of second player
            format_type: Cricket format (Test, ODI, T20I)
            visualization_paths: Dictionary mapping visualization types to file paths
            
        Returns:
            str: Markdown report content
        """
        try:
            if comparison is None:
                logger.error("Comparison data cannot be None")
                return None
            
            # Start building the report
            report = f"# {player1_name} vs {player2_name}: {format_type} Cricket Comparison\n\n"
            
            # Basic metrics section
            report += "## Basic Career Statistics\n\n"
            
            if 'basic_metrics' in comparison and comparison['basic_metrics']:
                report += "| Metric | " + player1_name + " | " + player2_name + " | Difference | Ratio |\n"
                report += "|--------|" + "-" * len(player1_name) + "|" + "-" * len(player2_name) + "|-----------|------|\n"
                
                for metric, values in comparison['basic_metrics'].items():
                    p1_value = values['player1']
                    p2_value = values['player2']
                    diff = values['difference']
                    ratio = values['ratio']
                    
                    report += f"| {metric} | {p1_value} | {p2_value} | {diff} | {ratio} |\n"
                
                report += "\n"
            
            # Impact scores and batting indices
            report += "## Performance Indices\n\n"
            
            if 'impact_scores' in comparison and 'batting_indices' in comparison:
                report += "### Batting Impact Scores\n\n"
                report += f"- {player1_name}: {comparison['impact_scores'].get(player1_name, 'N/A')}\n"
                report += f"- {player2_name}: {comparison['impact_scores'].get(player2_name, 'N/A')}\n"
                
                if comparison['impact_scores'].get('difference') is not None:
                    diff = comparison['impact_scores']['difference']
                    better_player = player1_name if diff > 0 else player2_name
                    report += f"- Difference: {abs(diff)} in favor of {better_player}\n\n"
                
                report += "### Batting Indices\n\n"
                report += f"- {player1_name}: {comparison['batting_indices'].get(player1_name, 'N/A')}\n"
                report += f"- {player2_name}: {comparison['batting_indices'].get(player2_name, 'N/A')}\n"
                
                if comparison['batting_indices'].get('difference') is not None:
                    diff = comparison['batting_indices']['difference']
                    better_player = player1_name if diff > 0 else player2_name
                    report += f"- Difference: {abs(diff)} in favor of {better_player}\n\n"
            
            # Performance by opposition
            report += "## Performance by Opposition\n\n"
            
            if 'by_opposition' in comparison:
                player1_opp = comparison['by_opposition'].get(player1_name)
                player2_opp = comparison['by_opposition'].get(player2_name)
                
                if player1_opp is not None:
                    report += f"### {player1_name}'s Performance by Opposition\n\n"
                    report += "| Opposition | Average |\n"
                    report += "|-----------|--------|\n"
                    
                    for _, row in player1_opp.iterrows():
                        report += f"| {row['Opposition']} | {row['Average']} |\n"
                    
                    report += "\n"
                
                if player2_opp is not None:
                    report += f"### {player2_name}'s Performance by Opposition\n\n"
                    report += "| Opposition | Average |\n"
                    report += "|-----------|--------|\n"
                    
                    for _, row in player2_opp.iterrows():
                        report += f"| {row['Opposition']} | {row['Average']} |\n"
                    
                    report += "\n"
            
            # Career progression
            report += "## Career Progression\n\n"
            
            if 'career_progression' in comparison:
                player1_prog = comparison['career_progression'].get(player1_name)
                player2_prog = comparison['career_progression'].get(player2_name)
                
                if player1_prog is not None:
                    report += f"### {player1_name}'s Career Progression\n\n"
                    report += "| Year | Average |\n"
                    report += "|------|--------|\n"
                    
                    for _, row in player1_prog.iterrows():
                        report += f"| {row.iloc[0]} | {row['Average']} |\n"
                    
                    report += "\n"
                
                if player2_prog is not None:
                    report += f"### {player2_name}'s Career Progression\n\n"
                    report += "| Year | Average |\n"
                    report += "|------|--------|\n"
                    
                    for _, row in player2_prog.iterrows():
                        report += f"| {row.iloc[0]} | {row['Average']} |\n"
                    
                    report += "\n"
            
            # Visualizations section
            if visualization_paths:
                report += "## Visualizations\n\n"
                
                for viz_type, path in visualization_paths.items():
                    # Convert to relative path if needed
                    filename = os.path.basename(path)
                    report += f"### {viz_type.replace('_', ' ').title()}\n\n"
                    report += f"![{viz_type}]({filename})\n\n"
            
            # Conclusion
            report += "## Conclusion\n\n"
            
            # Generate a simple conclusion based on the comparison
            if 'impact_scores' in comparison and 'batting_indices' in comparison:
                p1_impact = comparison['impact_scores'].get(player1_name, 0)
                p2_impact = comparison['impact_scores'].get(player2_name, 0)
                p1_index = comparison['batting_indices'].get(player1_name, 0)
                p2_index = comparison['batting_indices'].get(player2_name, 0)
                
                # Determine overall better player
                p1_points = 0
                p2_points = 0
                
                if p1_impact > p2_impact:
                    p1_points += 1
                else:
                    p2_points += 1
                
                if p1_index > p2_index:
                    p1_points += 1
                else:
                    p2_points += 1
                
                if 'basic_metrics' in comparison and comparison['basic_metrics']:
                    for metric, values in comparison['basic_metrics'].items():
                        if values['difference'] > 0:
                            p1_points += 1
                        else:
                            p2_points += 1
                
                better_player = player1_name if p1_points > p2_points else player2_name
                
                report += f"Based on the comprehensive analysis of various metrics, {better_player} demonstrates "
                report += f"superior overall performance in {format_type} cricket. However, both players have "
                report += "their unique strengths and have made significant contributions to cricket.\n\n"
                
                report += "This comparison highlights the importance of considering multiple metrics when "
                report += "evaluating cricket players, as different metrics can reveal different aspects of "
                report += "a player's performance and contribution to the game."
            
            return report
        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            return None


# Example usage
if __name__ == "__main__":
    from statsguru_query import StatguruQuery
    from statsguru_extractor import StatguruDataExtractor
    
    # Create queries for Sachin Tendulkar and Virat Kohli's Test batting statistics
    query = StatguruQuery()
    extractor = StatguruDataExtractor()
    
    # Tendulkar data
    tendulkar_html = query.build_query(
        format="test",
        analysis_type="batting",
        player="tendulkar"
    ).execute()
    
    tendulkar_data = extractor.extract_data_from_html(tendulkar_html)
    tendulkar_df = tendulkar_data['tables'][0] if tendulkar_data['tables'] else None
    
    # Kohli data
    kohli_html = query.build_query(
        format="test",
        analysis_type="batting",
        player="kohli"
    ).execute()
    
    kohli_data = extractor.extract_data_from_html(kohli_html)
    kohli_df = kohli_data['tables'][0] if kohli_data['tables'] else None
    
    # Compare players
    calculator = CricketStatsCalculator()
    
    if tendulkar_df is not None and kohli_df is not None:
        comparison = calculator.compare_batting_careers(
            tendulkar_df, kohli_df, "Sachin Tendulkar", "Virat Kohli"
        )
        
        # Create visualizations
        viz_paths = calculator.visualize_player_comparison(
            comparison, "Sachin Tendulkar", "Virat Kohli"
        )
        
        # Generate report
        report = calculator.generate_comparison_report(
            comparison, "Sachin Tendulkar", "Virat Kohli", "Test", viz_paths
        )
        
        # Print report preview
        if report:
            print(report[:500] + "...\n\n(Report truncated)")
    
    query.close()

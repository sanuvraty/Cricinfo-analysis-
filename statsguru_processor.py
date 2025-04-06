"""
Statsguru Data Processing Module

This module provides functionality to process and analyze cricket statistics data
extracted from Cricinfo's Statsguru. It includes functions for various cricket-specific
calculations and comparisons.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StatguruDataProcessor:
    """
    Processes and analyzes cricket statistics data from Statsguru.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        pass
    
    def calculate_batting_average(self, df: pd.DataFrame, runs_col: str = 'Runs', 
                                 innings_col: str = 'Innings', not_outs_col: str = 'NO') -> float:
        """
        Calculate batting average (runs / (innings - not outs))
        
        Args:
            df: DataFrame containing batting statistics
            runs_col: Column name for runs
            innings_col: Column name for innings
            not_outs_col: Column name for not outs
            
        Returns:
            float: Calculated batting average
        """
        try:
            if runs_col not in df.columns or innings_col not in df.columns:
                logger.error(f"Required columns missing for batting average calculation")
                return None
            
            total_runs = df[runs_col].sum()
            total_innings = df[innings_col].sum()
            
            # Handle not outs if the column exists
            total_not_outs = 0
            if not_outs_col in df.columns:
                total_not_outs = df[not_outs_col].sum()
            
            # Calculate average
            dismissals = total_innings - total_not_outs
            if dismissals > 0:
                average = total_runs / dismissals
            else:
                average = total_runs if total_runs > 0 else 0
                
            return round(average, 2)
        except Exception as e:
            logger.error(f"Error calculating batting average: {e}")
            return None
    
    def calculate_bowling_average(self, df: pd.DataFrame, runs_col: str = 'Runs', 
                                 wickets_col: str = 'Wickets') -> float:
        """
        Calculate bowling average (runs / wickets)
        
        Args:
            df: DataFrame containing bowling statistics
            runs_col: Column name for runs conceded
            wickets_col: Column name for wickets taken
            
        Returns:
            float: Calculated bowling average
        """
        try:
            if runs_col not in df.columns or wickets_col not in df.columns:
                logger.error(f"Required columns missing for bowling average calculation")
                return None
            
            total_runs = df[runs_col].sum()
            total_wickets = df[wickets_col].sum()
            
            if total_wickets > 0:
                average = total_runs / total_wickets
            else:
                average = float('inf')  # No wickets taken
                
            return round(average, 2)
        except Exception as e:
            logger.error(f"Error calculating bowling average: {e}")
            return None
    
    def calculate_strike_rate(self, df: pd.DataFrame, runs_col: str = 'Runs', 
                             balls_col: str = 'Balls_Faced') -> float:
        """
        Calculate batting strike rate (runs / balls * 100)
        
        Args:
            df: DataFrame containing batting statistics
            runs_col: Column name for runs
            balls_col: Column name for balls faced
            
        Returns:
            float: Calculated strike rate
        """
        try:
            if runs_col not in df.columns or balls_col not in df.columns:
                logger.error(f"Required columns missing for strike rate calculation")
                return None
            
            total_runs = df[runs_col].sum()
            total_balls = df[balls_col].sum()
            
            if total_balls > 0:
                strike_rate = (total_runs / total_balls) * 100
            else:
                strike_rate = 0
                
            return round(strike_rate, 2)
        except Exception as e:
            logger.error(f"Error calculating strike rate: {e}")
            return None
    
    def calculate_bowling_strike_rate(self, df: pd.DataFrame, balls_col: str = 'Balls', 
                                     wickets_col: str = 'Wickets') -> float:
        """
        Calculate bowling strike rate (balls / wickets)
        
        Args:
            df: DataFrame containing bowling statistics
            balls_col: Column name for balls bowled
            wickets_col: Column name for wickets taken
            
        Returns:
            float: Calculated bowling strike rate
        """
        try:
            if balls_col not in df.columns or wickets_col not in df.columns:
                logger.error(f"Required columns missing for bowling strike rate calculation")
                return None
            
            total_balls = df[balls_col].sum()
            total_wickets = df[wickets_col].sum()
            
            if total_wickets > 0:
                strike_rate = total_balls / total_wickets
            else:
                strike_rate = float('inf')  # No wickets taken
                
            return round(strike_rate, 2)
        except Exception as e:
            logger.error(f"Error calculating bowling strike rate: {e}")
            return None
    
    def calculate_economy_rate(self, df: pd.DataFrame, runs_col: str = 'Runs', 
                              balls_col: str = 'Balls') -> float:
        """
        Calculate bowling economy rate (runs / overs)
        
        Args:
            df: DataFrame containing bowling statistics
            runs_col: Column name for runs conceded
            balls_col: Column name for balls bowled
            
        Returns:
            float: Calculated economy rate
        """
        try:
            if runs_col not in df.columns or balls_col not in df.columns:
                logger.error(f"Required columns missing for economy rate calculation")
                return None
            
            total_runs = df[runs_col].sum()
            total_balls = df[balls_col].sum()
            
            if total_balls > 0:
                # Convert balls to overs (6 balls per over)
                overs = total_balls / 6
                economy_rate = total_runs / overs
            else:
                economy_rate = 0
                
            return round(economy_rate, 2)
        except Exception as e:
            logger.error(f"Error calculating economy rate: {e}")
            return None
    
    def calculate_batting_impact_score(self, df: pd.DataFrame) -> float:
        """
        Calculate batting impact score based on multiple factors
        
        Args:
            df: DataFrame containing batting statistics
            
        Returns:
            float: Calculated impact score
        """
        try:
            # Check for required columns
            required_cols = ['Runs', 'Strike_Rate', 'Average']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing columns for impact score: {missing_cols}")
                return None
            
            # Calculate basic metrics if not already present
            if 'Strike_Rate' not in df.columns and 'Runs' in df.columns and 'Balls_Faced' in df.columns:
                df['Strike_Rate'] = df.apply(
                    lambda row: (row['Runs'] / row['Balls_Faced'] * 100) if row['Balls_Faced'] > 0 else 0, 
                    axis=1
                )
            
            if 'Average' not in df.columns:
                avg = self.calculate_batting_average(df)
                if avg is not None:
                    df['Average'] = avg
            
            # Calculate impact score components
            # 1. Run contribution
            total_runs = df['Runs'].sum()
            run_impact = min(total_runs / 5000, 1) * 40  # Max 40 points for runs
            
            # 2. Average impact
            avg_impact = min(df['Average'].mean() / 50, 1) * 30  # Max 30 points for average
            
            # 3. Strike rate impact
            sr_impact = min(df['Strike_Rate'].mean() / 100, 1) * 20  # Max 20 points for SR
            
            # 4. Consistency impact (based on 50s and 100s if available)
            consistency_impact = 0
            if 'Hundreds' in df.columns and 'Fifties' in df.columns:
                innings = df['Innings'].sum() if 'Innings' in df.columns else len(df)
                if innings > 0:
                    hundreds = df['Hundreds'].sum()
                    fifties = df['Fifties'].sum()
                    milestone_ratio = (hundreds * 2 + fifties) / innings
                    consistency_impact = min(milestone_ratio / 0.5, 1) * 10  # Max 10 points
            
            # Calculate total impact score
            impact_score = run_impact + avg_impact + sr_impact + consistency_impact
            
            return round(impact_score, 2)
        except Exception as e:
            logger.error(f"Error calculating batting impact score: {e}")
            return None
    
    def calculate_bowling_impact_score(self, df: pd.DataFrame) -> float:
        """
        Calculate bowling impact score based on multiple factors
        
        Args:
            df: DataFrame containing bowling statistics
            
        Returns:
            float: Calculated impact score
        """
        try:
            # Check for required columns
            required_cols = ['Wickets', 'Average', 'Economy']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing columns for bowling impact score: {missing_cols}")
                return None
            
            # Calculate basic metrics if not already present
            if 'Average' not in df.columns:
                avg = self.calculate_bowling_average(df)
                if avg is not None:
                    df['Average'] = avg
            
            if 'Economy' not in df.columns and 'Runs' in df.columns and 'Balls' in df.columns:
                econ = self.calculate_economy_rate(df)
                if econ is not None:
                    df['Economy'] = econ
            
            # Calculate impact score components
            # 1. Wicket contribution
            total_wickets = df['Wickets'].sum()
            wicket_impact = min(total_wickets / 300, 1) * 40  # Max 40 points for wickets
            
            # 2. Average impact (lower is better)
            avg = df['Average'].mean()
            if avg > 0:
                avg_impact = min(40 / avg, 1) * 30  # Max 30 points for average
            else:
                avg_impact = 30  # Perfect average
            
            # 3. Economy impact (lower is better)
            econ = df['Economy'].mean()
            if econ > 0:
                econ_impact = min(6 / econ, 1) * 20  # Max 20 points for economy
            else:
                econ_impact = 20  # Perfect economy
            
            # 4. Five-wicket hauls impact
            five_wicket_impact = 0
            if 'FiveWickets' in df.columns:
                innings = df['Innings'].sum() if 'Innings' in df.columns else len(df)
                if innings > 0:
                    five_wickets = df['FiveWickets'].sum()
                    five_wicket_ratio = five_wickets / innings
                    five_wicket_impact = min(five_wicket_ratio / 0.1, 1) * 10  # Max 10 points
            
            # Calculate total impact score
            impact_score = wicket_impact + avg_impact + econ_impact + five_wicket_impact
            
            return round(impact_score, 2)
        except Exception as e:
            logger.error(f"Error calculating bowling impact score: {e}")
            return None
    
    def calculate_era_benchmark(self, player_df: pd.DataFrame, era_df: pd.DataFrame, 
                               metric: str = 'Average') -> Dict[str, float]:
        """
        Calculate performance relative to era benchmark
        
        Args:
            player_df: DataFrame containing player statistics
            era_df: DataFrame containing era statistics
            metric: Metric to compare (Average, Strike_Rate, etc.)
            
        Returns:
            dict: Dictionary with player metric, era metric, and relative performance
        """
        try:
            if metric not in player_df.columns or metric not in era_df.columns:
                logger.error(f"Metric {metric} not found in dataframes")
                return None
            
            player_metric = player_df[metric].mean()
            era_metric = era_df[metric].mean()
            
            # For batting metrics, higher is better
            batting_metrics = ['Runs', 'Average', 'Strike_Rate', 'Hundreds', 'Fifties']
            
            if metric in batting_metrics:
                relative_performance = player_metric / era_metric if era_metric > 0 else float('inf')
            else:
                # For bowling metrics, lower is better
                relative_performance = era_metric / player_metric if player_metric > 0 else float('inf')
            
            return {
                'player_metric': round(player_metric, 2),
                'era_metric': round(era_metric, 2),
                'relative_performance': round(relative_performance, 2)
            }
        except Exception as e:
            logger.error(f"Error calculating era benchmark: {e}")
            return None
    
    def calculate_weighted_performance(self, df: pd.DataFrame, weights: Dict[str, float]) -> float:
        """
        Calculate weighted performance metric
        
        Args:
            df: DataFrame containing statistics
            weights: Dictionary mapping metrics to weights
            
        Returns:
            float: Calculated weighted performance
        """
        try:
            weighted_sum = 0
            weight_total = sum(weights.values())
            
            for metric, weight in weights.items():
                if metric in df.columns:
                    # Normalize weight
                    norm_weight = weight / weight_total
                    
                    # Get metric value
                    metric_value = df[metric].mean()
                    
                    # Add to weighted sum
                    weighted_sum += metric_value * norm_weight
                else:
                    logger.warning(f"Metric {metric} not found in dataframe")
            
            return round(weighted_sum, 2)
        except Exception as e:
            logger.error(f"Error calculating weighted performance: {e}")
            return None
    
    def compare_players(self, player1_df: pd.DataFrame, player2_df: pd.DataFrame, 
                       metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare statistics between two players
        
        Args:
            player1_df: DataFrame containing first player's statistics
            player2_df: DataFrame containing second player's statistics
            metrics: List of metrics to compare
            
        Returns:
            dict: Dictionary with comparison results for each metric
        """
        try:
            if player1_df is None or player2_df is None:
                logger.error("Player dataframes cannot be None")
                return None
            
            # Default metrics if none provided
            if metrics is None:
                # Try to determine if batting or bowling data
                if 'Wickets' in player1_df.columns and 'Wickets' in player2_df.columns:
                    # Bowling data
                    metrics = ['Wickets', 'Average', 'Economy', 'Strike_Rate']
                else:
                    # Batting data
                    metrics = ['Runs', 'Average', 'Strike_Rate', 'Hundreds', 'Fifties']
            
            comparison = {}
            
            for metric in metrics:
                if metric in player1_df.columns and metric in player2_df.columns:
                    # Determine if sum or mean is appropriate
                    sum_metrics = ['Runs', 'Wickets', 'Matches', 'Innings', 'Hundreds', 'Fifties']
                    
                    if metric in sum_metrics:
                        p1_value = player1_df[metric].sum()
                        p2_value = player2_df[metric].sum()
                    else:
                        p1_value = player1_df[metric].mean()
                        p2_value = player2_df[metric].mean()
                    
                    # Calculate difference and ratio
                    difference = p1_value - p2_value
                    
                    # Avoid division by zero
                    if p2_value != 0:
                        ratio = p1_value / p2_value
                    else:
                        ratio = float('inf') if p1_value > 0 else 0
                    
                    comparison[metric] = {
                        'player1': round(p1_value, 2),
                        'player2': round(p2_value, 2),
                        'difference': round(difference, 2),
                        'ratio': round(ratio, 2)
                    }
                else:
                    logger.warning(f"Metric {metric} not found in both dataframes")
            
            return comparison
        except Exception as e:
            logger.error(f"Error comparing players: {e}")
            return None
    
    def calculate_career_progression(self, df: pd.DataFrame, metric: str = 'Average', 
                                    period: str = 'year') -> pd.DataFrame:
        """
        Calculate career progression over time
        
        Args:
            df: DataFrame containing statistics with date information
            metric: Metric to track over time
            period: Time period for grouping ('year', 'month', etc.)
            
        Returns:
            DataFrame: Career progression data
        """
        try:
            if 'Date' not in df.columns or metric not in df.columns:
                logger.error(f"Required columns missing for career progression")
                return None
            
            # Ensure Date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Group by time period
            if period == 'year':
                grouped = df.groupby(df['Date'].dt.year)
            elif period == 'month':
                grouped = df.groupby([df['Date'].dt.year, df['Date'].dt.month])
            else:
                logger.error(f"Unsupported period: {period}")
                return None
            
            # Determine if sum or mean is appropriate
            sum_metrics = ['Runs', 'Wickets', 'Matches', 'Innings', 'Hundreds', 'Fifties']
            
            if metric in sum_metrics:
                progression = grouped[metric].sum().reset_index()
            else:
                progression = grouped[metric].mean().reset_index()
            
            # Calculate cumulative metrics
            if metric in sum_metrics:
                progression['Cumulative'] = progression[metric].cumsum()
                
                # Calculate moving average if enough data points
                if len(progression) >= 3:
                    progression['Moving_Average'] = progression[metric].rolling(window=3, min_periods=1).mean()
            
            return progression
        except Exception as e:
            logger.error(f"Error calculating career progression: {e}")
            return None
    
    def calculate_performance_by_condition(self, df: pd.DataFrame, condition_col: str, 
                                         metric: str = 'Average') -> pd.DataFrame:
        """
        Calculate performance breakdown by condition
        
        Args:
            df: DataFrame containing statistics
            condition_col: Column to group by (e.g., 'Opposition', 'Ground')
            metric: Metric to analyze
            
        Returns:
            DataFrame: Performance by condition
        """
        try:
            if condition_col not in df.columns or metric not in df.columns:
                logger.error(f"Required columns missing for condition analysis")
                return None
            
            # Group by condition
            grouped = df.groupby(condition_col)
            
            # Determine if sum or mean is appropriate
            sum_metrics = ['Runs', 'Wickets', 'Matches', 'Innings', 'Hundreds', 'Fifties']
            
            if metric in sum_metrics:
                performance = grouped[metric].sum().reset_index()
            else:
                performance = grouped[metric].mean().reset_index()
            
            # Sort by metric value in descending order
            performance = performance.sort_values(by=metric, ascending=False)
            
            return performance
        except Exception as e:
            logger.error(f"Error calculating performance by condition: {e}")
            return None
    
    def calculate_batting_index(self, df: pd.DataFrame) -> float:
        """
        Calculate comprehensive batting index
        
        Args:
            df: DataFrame containing batting statistics
            
        Returns:
            float: Calculated batting index
        """
        try:
            # Check for required columns
            required_cols = ['Runs', 'Average', 'Strike_Rate', 'Innings']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing columns for batting index: {missing_cols}")
                return None
            
            # Component 1: Volume of runs (max 30 points)
            total_runs = df['Runs'].sum()
            run_component = min(total_runs / 10000, 1) * 30
            
            # Component 2: Batting average (max 25 points)
            avg = df['Average'].mean()
            avg_component = min(avg / 50, 1) * 25
            
            # Component 3: Strike rate (max 20 points)
            sr = df['Strike_Rate'].mean()
            sr_component = min(sr / 100, 1) * 20
            
            # Component 4: Consistency (max 15 points)
            consistency = 0
            if 'Hundreds' in df.columns and 'Fifties' in df.columns:
                innings = df['Innings'].sum()
                hundreds = df['Hundreds'].sum()
                fifties = df['Fifties'].sum()
                
                if innings > 0:
                    century_ratio = hundreds / innings
                    fifty_ratio = fifties / innings
                    consistency = (century_ratio * 10 + fifty_ratio * 5) * 15
                    consistency = min(consistency, 15)
            
            # Component 5: Longevity (max 10 points)
            innings = df['Innings'].sum()
            longevity = min(innings / 200, 1) * 10
            
            # Calculate total index
            batting_index = run_component + avg_component + sr_component + consistency + longevity
            
            return round(batting_index, 2)
        except Exception as e:
            logger.error(f"Error calculating batting index: {e}")
            return None


# Example usage
if __name__ == "__main__":
    from statsguru_query import StatguruQuery
    from statsguru_extractor import StatguruDataExtractor
    
    # Create a query for Sachin Tendulkar's Test batting statistics
    query = StatguruQuery()
    html_content = query.build_query(
        format="test",
        analysis_type="batting",
        player="tendulkar"
    ).execute()
    
    # Extract data
    extractor = StatguruDataExtractor()
    data = extractor.extract_data_from_html(html_content)
    
    # Process data
    processor = StatguruDataProcessor()
    
    if data['tables'] and len(data['tables']) > 0:
        df = data['tables'][0]
        
        # Calculate some metrics
        batting_avg = processor.calculate_batting_average(df)
        strike_rate = processor.calculate_strike_rate(df)
        impact_score = processor.calculate_batting_impact_score(df)
        
        print(f"Batting Average: {batting_avg}")
        print(f"Strike Rate: {strike_rate}")
        print(f"Impact Score: {impact_score}")
    
    query.close()

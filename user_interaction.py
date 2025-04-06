"""
User Interaction Module for Cricinfo Statsguru AI Agent

This module provides a natural language interface for users to interact with
the Cricinfo Statsguru AI Agent, allowing them to execute queries and perform
calculations on cricket statistics data.
"""

import re
import logging
import json
import os
from typing import Dict, List, Union, Optional, Tuple, Any

from statsguru_query import StatguruQuery
from statsguru_extractor import StatguruDataExtractor
from statsguru_processor import StatguruDataProcessor
from cricket_stats_calculator import CricketStatsCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CricinfoAgent:
    """
    Main agent class that handles user interactions and executes commands.
    """
    
    def __init__(self):
        """Initialize the Cricinfo Statsguru AI Agent."""
        self.query = StatguruQuery()
        self.extractor = StatguruDataExtractor()
        self.processor = StatguruDataProcessor()
        self.calculator = CricketStatsCalculator()
        
        # Create output directory
        self.output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cache for storing query results
        self.cache = {}
        
        logger.info("Cricinfo Statsguru AI Agent initialized")
    
    def process_command(self, command: str) -> Dict[str, Any]:
        """
        Process a natural language command from the user
        
        Args:
            command (str): User's natural language command
            
        Returns:
            dict: Response containing results and/or error messages
        """
        try:
            logger.info(f"Processing command: {command}")
            
            # Normalize command
            command = command.strip().lower()
            
            # Check for help command
            if "help" in command or "examples" in command:
                return self._get_help()
            
            # Extract command type and parameters
            command_type, params = self._parse_command(command)
            
            if command_type == "query":
                return self._execute_query(params)
            elif command_type == "compare":
                return self._compare_players(params)
            elif command_type == "calculate":
                return self._calculate_stats(params)
            elif command_type == "visualize":
                return self._create_visualization(params)
            elif command_type == "report":
                return self._generate_report(params)
            else:
                return {
                    "success": False,
                    "error": "Unknown command type. Try 'help' for available commands."
                }
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return {
                "success": False,
                "error": f"Error processing command: {str(e)}"
            }
    
    def _parse_command(self, command: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a natural language command to extract command type and parameters
        
        Args:
            command (str): User's natural language command
            
        Returns:
            tuple: Command type and parameters dictionary
        """
        # Initialize parameters
        params = {}
        
        # Check for query commands
        if any(x in command for x in ["search", "find", "get", "query", "show", "display"]):
            # Extract player name
            player_match = re.search(r"for\s+([a-z\s]+?)(?:\s+in|\s+against|\s+from|\s+during|\s+$)", command)
            if player_match:
                params["player"] = player_match.group(1).strip()
            
            # Extract format
            format_match = re.search(r"in\s+(test|odi|t20i|t20|all)(?:\s+|$)", command)
            if format_match:
                params["format"] = format_match.group(1).strip()
            else:
                params["format"] = "test"  # Default to Test format
            
            # Extract analysis type
            if "bowling" in command:
                params["analysis_type"] = "bowling"
            elif "fielding" in command:
                params["analysis_type"] = "fielding"
            else:
                params["analysis_type"] = "batting"  # Default to batting
            
            # Extract opposition
            opposition_match = re.search(r"against\s+([a-z\s]+?)(?:\s+in|\s+from|\s+during|\s+$)", command)
            if opposition_match:
                params["opposition"] = opposition_match.group(1).strip()
            
            # Extract date range
            date_range_match = re.search(r"from\s+(\d{4})\s+to\s+(\d{4})", command)
            if date_range_match:
                params["start_date"] = date_range_match.group(1).strip()
                params["end_date"] = date_range_match.group(2).strip()
            
            # Extract venue type
            if "home" in command:
                params["home_away"] = "home"
            elif "away" in command:
                params["home_away"] = "away"
            
            return "query", params
        
        # Check for comparison commands
        elif any(x in command for x in ["compare", "comparison", "versus", "vs"]):
            # Extract player names
            players_match = re.search(r"(compare|comparison|versus|vs)(?:\s+between)?\s+([a-z\s]+?)\s+and\s+([a-z\s]+?)(?:\s+in|\s+$)", command)
            if players_match:
                params["player1"] = players_match.group(2).strip()
                params["player2"] = players_match.group(3).strip()
            
            # Extract format
            format_match = re.search(r"in\s+(test|odi|t20i|t20|all)(?:\s+|$)", command)
            if format_match:
                params["format"] = format_match.group(1).strip()
            else:
                params["format"] = "test"  # Default to Test format
            
            # Extract analysis type
            if "bowling" in command:
                params["analysis_type"] = "bowling"
            else:
                params["analysis_type"] = "batting"  # Default to batting
            
            return "compare", params
        
        # Check for calculation commands
        elif any(x in command for x in ["calculate", "compute", "determine"]):
            # Extract calculation type
            if "impact" in command:
                params["calculation"] = "impact"
            elif "era" in command:
                params["calculation"] = "era_adjusted"
            elif "weighted" in command:
                params["calculation"] = "weighted"
            elif "match winning" in command:
                params["calculation"] = "match_winning"
            elif "batting index" in command:
                params["calculation"] = "batting_index"
            else:
                params["calculation"] = "basic"
            
            # Extract player name
            player_match = re.search(r"for\s+([a-z\s]+?)(?:\s+in|\s+$)", command)
            if player_match:
                params["player"] = player_match.group(1).strip()
            
            # Extract format
            format_match = re.search(r"in\s+(test|odi|t20i|t20|all)(?:\s+|$)", command)
            if format_match:
                params["format"] = format_match.group(1).strip()
            else:
                params["format"] = "test"  # Default to Test format
            
            return "calculate", params
        
        # Check for visualization commands
        elif any(x in command for x in ["visualize", "plot", "graph", "chart"]):
            # Extract visualization type
            if "career progression" in command:
                params["visualization"] = "career_progression"
            elif "opposition" in command:
                params["visualization"] = "by_opposition"
            elif "comparison" in command:
                params["visualization"] = "comparison"
            else:
                params["visualization"] = "basic"
            
            # Extract player name(s)
            players_match = re.search(r"for\s+([a-z\s]+?)(?:\s+and\s+([a-z\s]+?))?(?:\s+in|\s+$)", command)
            if players_match:
                params["player1"] = players_match.group(1).strip()
                if players_match.group(2):
                    params["player2"] = players_match.group(2).strip()
            
            # Extract format
            format_match = re.search(r"in\s+(test|odi|t20i|t20|all)(?:\s+|$)", command)
            if format_match:
                params["format"] = format_match.group(1).strip()
            else:
                params["format"] = "test"  # Default to Test format
            
            return "visualize", params
        
        # Check for report commands
        elif any(x in command for x in ["report", "generate report", "create report"]):
            # Extract player names
            players_match = re.search(r"for\s+([a-z\s]+?)\s+and\s+([a-z\s]+?)(?:\s+in|\s+$)", command)
            if players_match:
                params["player1"] = players_match.group(1).strip()
                params["player2"] = players_match.group(2).strip()
            
            # Extract format
            format_match = re.search(r"in\s+(test|odi|t20i|t20|all)(?:\s+|$)", command)
            if format_match:
                params["format"] = format_match.group(1).strip()
            else:
                params["format"] = "test"  # Default to Test format
            
            return "report", params
        
        # Default to query with minimal parameters
        else:
            return "query", {"format": "test", "analysis_type": "batting"}
    
    def _execute_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a query to Statsguru
        
        Args:
            params (dict): Query parameters
            
        Returns:
            dict: Query results
        """
        try:
            logger.info(f"Executing query with params: {params}")
            
            # Check cache first
            cache_key = json.dumps(params, sort_keys=True)
            if cache_key in self.cache:
                logger.info("Using cached results")
                return self.cache[cache_key]
            
            # Build and execute query
            for param, value in params.items():
                method_name = f"set_{param}"
                if hasattr(self.query.query_builder, method_name):
                    method = getattr(self.query.query_builder, method_name)
                    method(value)
            
            # Execute query
            html_content = self.query.execute()
            
            if not html_content:
                return {
                    "success": False,
                    "error": "Failed to retrieve data from Statsguru"
                }
            
            # Extract data
            data = self.extractor.extract_data_from_html(html_content)
            
            if not data['tables'] or len(data['tables']) == 0:
                return {
                    "success": False,
                    "error": "No data found for the given query parameters"
                }
            
            # Process results
            result = {
                "success": True,
                "tables": [df.to_dict(orient='records') for df in data['tables']],
                "player_ids": data['player_ids']
            }
            
            # Cache results
            self.cache[cache_key] = result
            
            return result
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {
                "success": False,
                "error": f"Error executing query: {str(e)}"
            }
    
    def _compare_players(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare statistics between two players
        
        Args:
            params (dict): Comparison parameters
            
        Returns:
            dict: Comparison results
        """
        try:
            logger.info(f"Comparing players with params: {params}")
            
            # Check required parameters
            if 'player1' not in params or 'player2' not in params:
                return {
                    "success": False,
                    "error": "Player names are required for comparison"
                }
            
            # Set default format if not provided
            if 'format' not in params:
                params['format'] = 'test'
            
            # Set default analysis type if not provided
            if 'analysis_type' not in params:
                params['analysis_type'] = 'batting'
            
            # Query data for player 1
            player1_params = {
                'player': params['player1'],
                'format': params['format'],
                'analysis_type': params['analysis_type']
            }
            player1_result = self._execute_query(player1_params)
            
            if not player1_result['success']:
                return {
                    "success": False,
                    "error": f"Error retrieving data for {params['player1']}: {player1_result.get('error', 'Unknown error')}"
                }
            
            # Query data for player 2
            player2_params = {
                'player': params['player2'],
                'format': params['format'],
                'analysis_type': params['analysis_type']
            }
            player2_result = self._execute_query(player2_params)
            
            if not player2_result['success']:
                return {
                    "success": False,
                    "error": f"Error retrieving data for {params['player2']}: {player2_result.get('error', 'Unknown error')}"
                }
            
            # Convert dict records back to DataFrames
            import pandas as pd
            player1_df = pd.DataFrame(player1_result['tables'][0])
            player2_df = pd.DataFrame(player2_result['tables'][0])
            
            # Compare players
            if params['analysis_type'] == 'batting':
                comparison = self.calculator.compare_batting_careers(
                    player1_df, player2_df, params['player1'], params['player2']
                )
            else:
                # For bowling comparison, use the basic comparison function
                comparison = self.processor.compare_players(player1_df, player2_df)
            
            if not comparison:
                return {
                    "success": False,
                    "error": "Failed to compare players"
                }
            
            # Create visualizations
            viz_paths = self.calculator.visualize_player_comparison(
                comparison, params['player1'], params['player2'], self.output_dir
            )
            
            # Return results
            return {
                "success": True,
                "comparison": comparison,
                "visualizations": viz_paths
            }
        except Exception as e:
            logger.error(f"Error comparing players: {e}")
            return {
                "success": False,
                "error": f"Error comparing players: {str(e)}"
            }
    
    def _calculate_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistics for a player
        
        Args:
            params (dict): Calculation parameters
            
        Returns:
            dict: Calculation results
        """
        try:
            logger.info(f"Calculating stats with params: {params}")
            
            # Check required parameters
            if 'player' not in params:
                return {
                    "success": False,
                    "error": "Player name is required for calculation"
                }
            
            # Set default format if not provided
            if 'format' not in params:
                params['format'] = 'test'
            
            # Set default calculation if not provided
            if 'calculation' not in params:
                params['calculation'] = 'basic'
            
            # Query data for player
            player_params = {
                'player': params['player'],
                'format': params['format'],
                'analysis_type': 'batting' if params['calculation'] != 'bowling_stats' else 'bowling'
            }
            player_result = self._execute_query(player_params)
            
            if not player_result['success']:
                return {
                    "success": False,
                    "error": f"Error retrieving data for {params['player']}: {player_result.get('error', 'Unknown error')}"
                }
            
            # Convert dict records back to DataFrame
            import pandas as pd
            player_df = pd.DataFrame(player_result['tables'][0])
            
            # Perform calculation based on type
            calculation_type = params['calculation']
            result = {}
            
            if calculation_type == 'basic':
                # Basic statistics
                if 'Runs' in player_df.columns:
                    result['total_runs'] = player_df['Runs'].sum()
                if 'Average' in player_df.columns:
                    result['average'] = player_df['Average'].mean()
                if 'Strike_Rate' in player_df.columns:
                    result['strike_rate'] = player_df['Strike_Rate'].mean()
                if 'Hundreds' in player_df.columns:
                    result['hundreds'] = player_df['Hundreds'].sum()
                if 'Fifties' in player_df.columns:
                    result['fifties'] = player_df['Fifties'].sum()
                if 'Wickets' in player_df.columns:
                    result['total_wickets'] = player_df['Wickets'].sum()
                if 'Economy' in player_df.columns:
                    result['economy'] = player_df['Economy'].mean()
            
            elif calculation_type == 'impact':
                # Impact score
                if 'Wickets' in player_df.columns:
                    result['bowling_impact'] = self.processor.calculate_bowling_impact_score(player_df)
                else:
                    result['batting_impact'] = self.processor.calculate_batting_impact_score(player_df)
            
            elif calculation_type == 'era_adjusted':
                # Era-adjusted stats
                # For this, we need to query era data
                era_params = {
                    'format': params['format'],
                    'analysis_type': player_params['analysis_type']
                }
                era_result = self._execute_query(era_params)
                
                if not era_result['success']:
                    return {
                        "success": False,
                        "error": f"Error retrieving era data: {era_result.get('error', 'Unknown error')}"
                    }
                
                era_df = pd.DataFrame(era_result['tables'][0])
                
                # Calculate era-adjusted stats
                result['era_adjusted'] = self.calculator.calculate_era_adjusted_stats(player_df, era_df)
            
            elif calculation_type == 'weighted':
                # Weighted rating
                result['weighted_rating'] = self.calculator.calculate_weighted_batting_rating(player_df)
            
            elif calculation_type == 'match_winning':
                # Match-winning contribution
                if 'Result' in player_df.columns:
                    result['match_winning'] = self.calculator.calculate_match_winning_contribution(player_df)
                else:
                    result['error'] = "Match result information not available for match-winning calculation"
            
            elif calculation_type == 'batting_index':
                # Batting index
                result['batting_index'] = self.processor.calculate_batting_index(player_df)
            
            # Return results
            return {
                "success": True,
                "player": params['player'],
                "format": params['format'],
                "calculation_type": calculation_type,
                "results": result
            }
        except Exception as e:
            logger.error(f"Error calculating stats: {e}")
            return {
                "success": False,
                "error": f"Error calculating stats: {str(e)}"
            }
    
    def _create_visualization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create visualizations for player statistics
        
        Args:
            params (dict): Visualization parameters
            
        Returns:
            dict: Visualization results
        """
        try:
            logger.info(f"Creating visualization with params: {params}")
            
            # Check required parameters
            if 'player1' not in params:
                return {
                    "success": False,
                    "error": "At least one player name is required for visualization"
                }
            
            # Set default format if not provided
            if 'format' not in params:
                params['format'] = 'test'
            
            # Set default visualization type if not provided
            if 'visualization' not in params:
                params['visualization'] = 'basic'
            
            # Query data for player 1
            player1_params = {
                'player': params['player1'],
                'format': params['format'],
                'analysis_type': 'batting'
            }
            player1_result = self._execute_query(player1_params)
            
            if not player1_result['success']:
                return {
                    "success": False,
                    "error": f"Error retrieving data for {params['player1']}: {player1_result.get('error', 'Unknown error')}"
                }
            
            # Convert dict records back to DataFrame
            import pandas as pd
            player1_df = pd.DataFrame(player1_result['tables'][0])
            
            # Check if we need to compare two players
            player2_df = None
            if 'player2' in params:
                player2_params = {
                    'player': params['player2'],
                    'format': params['format'],
                    'analysis_type': 'batting'
                }
                player2_result = self._execute_query(player2_params)
                
                if not player2_result['success']:
                    return {
                        "success": False,
                        "error": f"Error retrieving data for {params['player2']}: {player2_result.get('error', 'Unknown error')}"
                    }
                
                player2_df = pd.DataFrame(player2_result['tables'][0])
            
            # Create visualization based on type
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            
            viz_type = params['visualization']
            viz_path = None
            
            if viz_type == 'basic':
                # Basic statistics visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                
                metrics = ['Runs', 'Average', 'Strike_Rate', 'Hundreds', 'Fifties']
                available_metrics = [m for m in metrics if m in player1_df.columns]
                
                values = []
                for metric in available_metrics:
                    if metric in ['Runs', 'Hundreds', 'Fifties']:
                        values.append(player1_df[metric].sum())
                    else:
                        values.append(player1_df[metric].mean())
                
                ax.bar(available_metrics, values, color='skyblue')
                ax.set_title(f"{params['player1']}'s {params['format'].upper()} Batting Statistics")
                ax.set_ylabel('Value')
                
                # Add value labels on top of bars
                for i, v in enumerate(values):
                    ax.text(i, v + 0.1, str(round(v, 2)), ha='center')
                
                plt.tight_layout()
                
                # Save figure
                viz_path = os.path.join(self.output_dir, f"{params['player1'].replace(' ', '_')}_basic_stats.png")
                plt.savefig(viz_path)
                plt.close(fig)
            
            elif viz_type == 'career_progression':
                # Career progression visualization
                if 'Date' not in player1_df.columns:
                    return {
                        "success": False,
                        "error": "Date information not available for career progression visualization"
                    }
                
                # Calculate career progression
                progression = self.processor.calculate_career_progression(player1_df, 'Average', 'year')
                
                if progression is None:
                    return {
                        "success": False,
                        "error": "Failed to calculate career progression"
                    }
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.plot(progression.iloc[:, 0], progression['Average'], marker='o', linestyle='-', color='blue')
                
                if 'Cumulative' in progression.columns:
                    ax2 = ax.twinx()
                    ax2.plot(progression.iloc[:, 0], progression['Cumulative'], marker='s', linestyle='--', color='red')
                    ax2.set_ylabel('Cumulative Runs', color='red')
                
                ax.set_title(f"{params['player1']}'s Career Progression in {params['format'].upper()}")
                ax.set_xlabel('Year')
                ax.set_ylabel('Batting Average', color='blue')
                
                plt.tight_layout()
                
                # Save figure
                viz_path = os.path.join(self.output_dir, f"{params['player1'].replace(' ', '_')}_career_progression.png")
                plt.savefig(viz_path)
                plt.close(fig)
            
            elif viz_type == 'by_opposition':
                # Performance by opposition visualization
                if 'Opposition' not in player1_df.columns:
                    return {
                        "success": False,
                        "error": "Opposition information not available for visualization"
                    }
                
                # Calculate performance by opposition
                performance = self.processor.calculate_performance_by_condition(player1_df, 'Opposition', 'Average')
                
                if performance is None:
                    return {
                        "success": False,
                        "error": "Failed to calculate performance by opposition"
                    }
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                ax.bar(performance['Opposition'], performance['Average'], color='skyblue')
                ax.set_title(f"{params['player1']}'s Performance by Opposition in {params['format'].upper()}")
                ax.set_xlabel('Opposition')
                ax.set_ylabel('Batting Average')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels on top of bars
                for i, v in enumerate(performance['Average']):
                    ax.text(i, v + 0.1, str(round(v, 2)), ha='center')
                
                plt.tight_layout()
                
                # Save figure
                viz_path = os.path.join(self.output_dir, f"{params['player1'].replace(' ', '_')}_by_opposition.png")
                plt.savefig(viz_path)
                plt.close(fig)
            
            elif viz_type == 'comparison' and player2_df is not None:
                # Player comparison visualization
                comparison = self.processor.compare_players(player1_df, player2_df)
                
                if comparison is None:
                    return {
                        "success": False,
                        "error": "Failed to compare players"
                    }
                
                fig, ax = plt.subplots(figsize=(12, 8))
                
                metrics = list(comparison.keys())
                player1_values = [comparison[m]['player1'] for m in metrics]
                player2_values = [comparison[m]['player2'] for m in metrics]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                ax.bar(x - width/2, player1_values, width, label=params['player1'])
                ax.bar(x + width/2, player2_values, width, label=params['player2'])
                
                ax.set_xticks(x)
                ax.set_xticklabels(metrics, rotation=45)
                ax.legend()
                
                ax.set_title(f"Comparison: {params['player1']} vs {params['player2']} in {params['format'].upper()}")
                ax.set_ylabel('Value')
                
                plt.tight_layout()
                
                # Save figure
                viz_path = os.path.join(self.output_dir, f"{params['player1'].replace(' ', '_')}_vs_{params['player2'].replace(' ', '_')}.png")
                plt.savefig(viz_path)
                plt.close(fig)
            
            # Return results
            if viz_path:
                return {
                    "success": True,
                    "visualization_type": viz_type,
                    "visualization_path": viz_path
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create visualization"
                }
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return {
                "success": False,
                "error": f"Error creating visualization: {str(e)}"
            }
    
    def _generate_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison report
        
        Args:
            params (dict): Report parameters
            
        Returns:
            dict: Report results
        """
        try:
            logger.info(f"Generating report with params: {params}")
            
            # Check required parameters
            if 'player1' not in params or 'player2' not in params:
                return {
                    "success": False,
                    "error": "Two player names are required for report generation"
                }
            
            # Set default format if not provided
            if 'format' not in params:
                params['format'] = 'test'
            
            # First, compare the players
            comparison_params = {
                'player1': params['player1'],
                'player2': params['player2'],
                'format': params['format'],
                'analysis_type': 'batting'
            }
            comparison_result = self._compare_players(comparison_params)
            
            if not comparison_result['success']:
                return {
                    "success": False,
                    "error": f"Error comparing players: {comparison_result.get('error', 'Unknown error')}"
                }
            
            # Generate report
            report = self.calculator.generate_comparison_report(
                comparison_result['comparison'],
                params['player1'],
                params['player2'],
                params['format'],
                comparison_result.get('visualizations')
            )
            
            if not report:
                return {
                    "success": False,
                    "error": "Failed to generate report"
                }
            
            # Save report to file
            report_path = os.path.join(
                self.output_dir, 
                f"{params['player1'].replace(' ', '_')}_vs_{params['player2'].replace(' ', '_')}_report.md"
            )
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            # Return results
            return {
                "success": True,
                "report_path": report_path,
                "report_content": report,
                "visualizations": comparison_result.get('visualizations')
            }
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                "success": False,
                "error": f"Error generating report: {str(e)}"
            }
    
    def _get_help(self) -> Dict[str, Any]:
        """
        Get help information and examples
        
        Returns:
            dict: Help information
        """
        help_text = """
        # Cricinfo Statsguru AI Agent Help
        
        This agent allows you to query Cricinfo's Statsguru, extract cricket statistics data,
        and perform calculations and comparisons.
        
        ## Available Commands
        
        ### Query Commands
        - "Get batting statistics for Sachin Tendulkar in test cricket"
        - "Show bowling figures for Dale Steyn against India"
        - "Find batting stats for Virat Kohli in ODIs from 2015 to 2020"
        
        ### Comparison Commands
        - "Compare Sachin Tendulkar and Virat Kohli in test cricket"
        - "Compare bowling stats of James Anderson and Dale Steyn"
        
        ### Calculation Commands
        - "Calculate batting impact score for Virat Kohli in test cricket"
        - "Compute era-adjusted stats for Don Bradman"
        - "Determine match-winning contributions for Ricky Ponting"
        
        ### Visualization Commands
        - "Visualize career progression for Sachin Tendulkar"
        - "Plot batting stats by opposition for Virat Kohli"
        - "Create comparison chart for Sachin Tendulkar and Virat Kohli"
        
        ### Report Commands
        - "Generate report for Sachin Tendulkar and Virat Kohli in test cricket"
        - "Create comprehensive comparison report for Ricky Ponting and Steve Smith"
        
        ## Tips
        - Specify the cricket format (test, odi, t20i) for more accurate results
        - For player comparisons, use "and" between player names
        - For date ranges, use "from [year] to [year]" format
        """
        
        return {
            "success": True,
            "help": help_text
        }
    
    def close(self):
        """Close the query session."""
        self.query.close()


# Example usage
if __name__ == "__main__":
    agent = CricinfoAgent()
    
    # Example commands
    commands = [
        "Get batting statistics for Sachin Tendulkar in test cricket",
        "Compare Sachin Tendulkar and Virat Kohli in test cricket",
        "Calculate batting impact score for Virat Kohli in test cricket",
        "Visualize career progression for Sachin Tendulkar",
        "Generate report for Sachin Tendulkar and Virat Kohli in test cricket"
    ]
    
    for command in commands:
        print(f"\nExecuting command: {command}")
        result = agent.process_command(command)
        print(f"Success: {result['success']}")
        if not result['success']:
            print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            print("Command executed successfully")
    
    agent.close()

"""
Statsguru Query Interface Module

This module provides functionality to construct and execute queries to Cricinfo's Statsguru.
It handles URL construction, parameter validation, and query execution.
"""

import requests
from urllib.parse import urlencode
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StatguruQueryBuilder:
    """
    Builds query URLs for Cricinfo's Statsguru based on user parameters.
    """
    BASE_URL = "https://stats.espncricinfo.com/ci/engine/stats/index.html"
    
    # Cricket format mapping
    FORMAT_MAP = {
        "test": 1,
        "odi": 2,
        "t20i": 3,
        "all": 11,
        "t20": 6,
        "womens_t20i": 8,
        "womens_test": 9,
        "womens_odi": 10
    }
    
    # Analysis type mapping
    ANALYSIS_MAP = {
        "batting": "batting",
        "bowling": "bowling",
        "fielding": "fielding",
        "allround": "all-round",
        "partnership": "fow-summary",
        "team": "team",
        "umpire": "umpire"
    }
    
    def __init__(self):
        """Initialize with empty parameters dictionary."""
        self.params = {}
    
    def set_format(self, format_type):
        """
        Set cricket format (Tests, ODIs, T20Is, etc.)
        
        Args:
            format_type (str): Cricket format (test, odi, t20i, etc.)
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        format_code = self.FORMAT_MAP.get(format_type.lower())
        if format_code:
            self.params["class"] = format_code
        else:
            logger.warning(f"Unknown format: {format_type}. Using Test as default.")
            self.params["class"] = 1
        return self
    
    def set_analysis_type(self, analysis_type):
        """
        Set analysis type (batting, bowling, etc.)
        
        Args:
            analysis_type (str): Type of analysis
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        analysis_code = self.ANALYSIS_MAP.get(analysis_type.lower())
        if analysis_code:
            self.params["type"] = analysis_code
        else:
            logger.warning(f"Unknown analysis type: {analysis_type}. Using batting as default.")
            self.params["type"] = "batting"
        return self
    
    def set_team(self, team):
        """
        Set team filter
        
        Args:
            team (str): Team name
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        self.params["team_id"] = team
        return self
    
    def set_opposition(self, opposition):
        """
        Set opposition team filter
        
        Args:
            opposition (str): Opposition team name
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        self.params["opposition"] = opposition
        return self
    
    def set_host(self, host):
        """
        Set host country filter
        
        Args:
            host (str): Host country
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        self.params["host"] = host
        return self
    
    def set_ground(self, ground):
        """
        Set ground/venue filter
        
        Args:
            ground (str): Ground/venue name
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        self.params["ground"] = ground
        return self
    
    def set_date_range(self, start_date, end_date):
        """
        Set date range filter
        
        Args:
            start_date (str): Start date in format DD MMM YYYY
            end_date (str): End date in format DD MMM YYYY
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        if start_date:
            self.params["spanmin1"] = start_date
        if end_date:
            self.params["spanmax1"] = end_date
        return self
    
    def set_season(self, season):
        """
        Set season filter
        
        Args:
            season (str): Season (e.g., 2019/20)
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        self.params["season"] = season
        return self
    
    def set_player(self, player_name):
        """
        Set player name filter
        
        Args:
            player_name (str): Player name
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        self.params["player_involve"] = player_name
        return self
    
    def set_template(self, template):
        """
        Set output template
        
        Args:
            template (str): Output template (results, innings, etc.)
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        self.params["template"] = template
        return self
    
    def set_result(self, result):
        """
        Set match result filter
        
        Args:
            result (str): Match result (won, lost, draw, tie)
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        if result.lower() in ["won", "win"]:
            self.params["result"] = 1
        elif result.lower() in ["lost", "lose"]:
            self.params["result"] = 2
        elif result.lower() == "tie":
            self.params["result"] = 3
        elif result.lower() in ["draw", "drawn"]:
            self.params["result"] = 4
        else:
            logger.warning(f"Unknown result: {result}. Ignoring.")
        return self
    
    def set_home_away(self, location):
        """
        Set home/away filter
        
        Args:
            location (str): Location (home, away, neutral)
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        if location.lower() == "home":
            self.params["home_or_away"] = 1
        elif location.lower() == "away":
            self.params["home_or_away"] = 2
        elif location.lower() == "neutral":
            self.params["home_or_away"] = 3
        else:
            logger.warning(f"Unknown location: {location}. Ignoring.")
        return self
    
    def set_advanced_filter(self, filter_name, filter_value):
        """
        Set custom advanced filter
        
        Args:
            filter_name (str): Filter parameter name
            filter_value (str): Filter parameter value
            
        Returns:
            StatguruQueryBuilder: Self for method chaining
        """
        self.params[filter_name] = filter_value
        return self
    
    def build_url(self):
        """
        Build the final URL with all parameters
        
        Returns:
            str: Complete URL for the Statsguru query
        """
        # Convert parameters to URL format
        query_parts = []
        for key, value in self.params.items():
            query_parts.append(f"{key}={value}")
        
        query_string = ";".join(query_parts)
        url = f"{self.BASE_URL}?{query_string}"
        
        logger.info(f"Built URL: {url}")
        return url


class StatguruQueryExecutor:
    """
    Executes queries to Cricinfo's Statsguru and handles responses.
    """
    
    def __init__(self, headers=None):
        """
        Initialize with optional custom headers.
        
        Args:
            headers (dict, optional): Custom HTTP headers
        """
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        self.session = requests.Session()
    
    def execute_query(self, url, max_retries=3, retry_delay=2):
        """
        Execute a query to Statsguru
        
        Args:
            url (str): Query URL
            max_retries (int, optional): Maximum number of retry attempts
            retry_delay (int, optional): Delay between retries in seconds
            
        Returns:
            str: HTML content of the response
        """
        import time
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Executing query: {url}")
                response = self.session.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logger.error(f"Error executing query (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries reached. Query failed.")
                    return None
    
    def close(self):
        """Close the session."""
        self.session.close()


class StatguruQuery:
    """
    Main interface for constructing and executing Statsguru queries.
    """
    
    def __init__(self):
        """Initialize with query builder and executor."""
        self.query_builder = StatguruQueryBuilder()
        self.query_executor = StatguruQueryExecutor()
    
    def build_query(self, **kwargs):
        """
        Build a query with the provided parameters
        
        Args:
            **kwargs: Query parameters
            
        Returns:
            StatguruQuery: Self for method chaining
        """
        # Reset query builder
        self.query_builder = StatguruQueryBuilder()
        
        # Apply parameters
        for param, value in kwargs.items():
            method_name = f"set_{param}"
            if hasattr(self.query_builder, method_name):
                method = getattr(self.query_builder, method_name)
                method(value)
            else:
                logger.warning(f"Unknown parameter: {param}. Using as advanced filter.")
                self.query_builder.set_advanced_filter(param, value)
        
        return self
    
    def execute(self):
        """
        Execute the built query
        
        Returns:
            str: HTML content of the response
        """
        url = self.query_builder.build_url()
        return self.query_executor.execute_query(url)
    
    def close(self):
        """Close the query executor session."""
        self.query_executor.close()


# Example usage
if __name__ == "__main__":
    # Create a query for Sachin Tendulkar's Test batting statistics
    query = StatguruQuery()
    html_content = query.build_query(
        format="test",
        analysis_type="batting",
        player="tendulkar"
    ).execute()
    
    # Print the first 100 characters of the response
    if html_content:
        print(f"Response received. Length: {len(html_content)}")
        print(f"Preview: {html_content[:100]}...")
    else:
        print("Query failed.")
    
    query.close()

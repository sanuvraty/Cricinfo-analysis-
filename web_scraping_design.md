# Web Scraping Solution Design for Cricinfo Statsguru

## Overview
This document outlines the design for a web scraping solution to extract cricket statistics data from Cricinfo's Statsguru. The solution will enable users to query Statsguru and perform calculations on the extracted data.

## Architecture

### 1. Components
- **Query Builder**: Constructs URLs for Statsguru queries based on user input
- **Web Scraper**: Handles HTTP requests and extracts HTML content
- **Data Parser**: Extracts structured data from HTML responses
- **Data Processor**: Performs calculations and transformations on extracted data
- **User Interface**: Accepts user commands and presents results

### 2. Technologies
- **Python**: Primary programming language
- **Requests**: For making HTTP requests
- **BeautifulSoup**: For parsing HTML and extracting data
- **Pandas**: For data manipulation and analysis
- **Matplotlib/Seaborn**: For data visualization
- **Selenium** (optional): For handling dynamic content if needed

## Implementation Details

### 1. Query Builder
```python
class StatguruQueryBuilder:
    BASE_URL = "https://stats.espncricinfo.com/ci/engine/stats/index.html"
    
    def __init__(self):
        self.params = {}
    
    def set_format(self, format_type):
        """Set cricket format (Tests, ODIs, T20Is)"""
        format_map = {
            "test": 1,
            "odi": 2,
            "t20i": 3,
            "all": 11
        }
        self.params["class"] = format_map.get(format_type.lower(), 1)
        return self
    
    def set_analysis_type(self, analysis_type):
        """Set analysis type (batting, bowling, etc.)"""
        self.params["type"] = analysis_type.lower()
        return self
    
    def set_player(self, player_id):
        """Set player for individual analysis"""
        self.params["player_id"] = player_id
        return self
    
    def set_team(self, team):
        """Set team filter"""
        self.params["team"] = team
        return self
    
    def set_opposition(self, opposition):
        """Set opposition team filter"""
        self.params["opposition"] = opposition
        return self
    
    def set_date_range(self, start_date, end_date):
        """Set date range filter"""
        self.params["spanmin1"] = start_date
        self.params["spanmax1"] = end_date
        return self
    
    def build_url(self):
        """Build the final URL with all parameters"""
        query_parts = []
        for key, value in self.params.items():
            query_parts.append(f"{key}={value}")
        
        query_string = ";".join(query_parts)
        return f"{self.BASE_URL}?{query_string}"
```

### 2. Web Scraper
```python
class StatguruScraper:
    def __init__(self, headers=None):
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def fetch_page(self, url):
        """Fetch HTML content from URL"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching page: {e}")
            return None
    
    def extract_tables(self, html_content):
        """Extract tables from HTML content"""
        if not html_content:
            return []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table', class_='engineTable')
        return tables
```

### 3. Data Parser
```python
class StatguruParser:
    def parse_table(self, table):
        """Parse HTML table into pandas DataFrame"""
        if not table:
            return None
        
        # Extract headers
        headers = []
        header_row = table.find('thead').find_all('th')
        for th in header_row:
            headers.append(th.text.strip())
        
        # Extract data rows
        rows = []
        data_rows = table.find('tbody').find_all('tr')
        for tr in data_rows:
            row = []
            cells = tr.find_all(['td', 'th'])
            for cell in cells:
                row.append(cell.text.strip())
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)
        return df
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        if df is None or df.empty:
            return df
        
        # Convert numeric columns
        numeric_columns = ['Runs', 'Balls', 'Mins', '4s', '6s', 'SR', 'Ave', 'Inns', 'Mat']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
```

### 4. Data Processor
```python
class StatguruProcessor:
    def calculate_batting_impact(self, df):
        """Calculate batting impact score"""
        if df is None or df.empty:
            return df
        
        if all(col in df.columns for col in ['Runs', 'SR', 'Ave']):
            # Example formula for batting impact
            df['Impact'] = df['Runs'] * df['SR'] / 100 * df['Ave'] / 50
        
        return df
    
    def calculate_era_benchmark(self, player_df, era_df):
        """Calculate performance relative to era benchmark"""
        if player_df is None or era_df is None:
            return None
        
        # Example calculation
        player_avg = player_df['Ave'].mean()
        era_avg = era_df['Ave'].mean()
        
        return {
            'player_avg': player_avg,
            'era_avg': era_avg,
            'relative_performance': player_avg / era_avg if era_avg > 0 else 0
        }
    
    def compare_players(self, player1_df, player2_df):
        """Compare statistics between two players"""
        if player1_df is None or player2_df is None:
            return None
        
        comparison = {}
        metrics = ['Runs', 'Ave', 'SR', 'Inns', 'Mat']
        
        for metric in metrics:
            if metric in player1_df.columns and metric in player2_df.columns:
                p1_value = player1_df[metric].sum() if metric == 'Runs' else player1_df[metric].mean()
                p2_value = player2_df[metric].sum() if metric == 'Runs' else player2_df[metric].mean()
                comparison[metric] = {
                    'player1': p1_value,
                    'player2': p2_value,
                    'difference': p1_value - p2_value,
                    'ratio': p1_value / p2_value if p2_value > 0 else float('inf')
                }
        
        return comparison
```

### 5. User Interface
```python
class StatguruAgent:
    def __init__(self):
        self.query_builder = StatguruQueryBuilder()
        self.scraper = StatguruScraper()
        self.parser = StatguruParser()
        self.processor = StatguruProcessor()
    
    def execute_query(self, query_params):
        """Execute a query based on parameters"""
        # Build query URL
        for param, value in query_params.items():
            method_name = f"set_{param}"
            if hasattr(self.query_builder, method_name):
                method = getattr(self.query_builder, method_name)
                method(value)
        
        url = self.query_builder.build_url()
        
        # Fetch and parse data
        html_content = self.scraper.fetch_page(url)
        tables = self.scraper.extract_tables(html_content)
        
        if not tables:
            return None
        
        # Parse main table
        df = self.parser.parse_table(tables[0])
        df = self.parser.clean_data(df)
        
        return df
    
    def compare_players(self, player1_name, player2_name, format_type="test"):
        """Compare statistics between two players"""
        # Execute queries for both players
        player1_df = self.execute_query({
            "format": format_type,
            "analysis_type": "batting",
            "player": player1_name
        })
        
        player2_df = self.execute_query({
            "format": format_type,
            "analysis_type": "batting",
            "player": player2_name
        })
        
        # Process comparison
        comparison = self.processor.compare_players(player1_df, player2_df)
        
        return comparison
    
    def calculate_custom_metric(self, player_name, metric_function, format_type="test"):
        """Calculate custom metric for a player"""
        # Execute query for player
        player_df = self.execute_query({
            "format": format_type,
            "analysis_type": "batting",
            "player": player_name
        })
        
        if player_df is None:
            return None
        
        # Apply custom metric function
        result = metric_function(player_df)
        
        return result
```

## Error Handling and Resilience

1. **Rate Limiting**: Implement delays between requests to avoid being blocked
2. **Retry Mechanism**: Retry failed requests with exponential backoff
3. **Error Logging**: Log errors for debugging and monitoring
4. **Data Validation**: Validate extracted data to ensure integrity
5. **Graceful Degradation**: Return partial results when possible instead of failing completely

## Extensibility

The solution is designed to be extensible in the following ways:
1. **New Query Types**: Add methods to QueryBuilder for new query parameters
2. **Custom Calculations**: Support custom calculation functions via the processor
3. **Output Formats**: Add support for different output formats (CSV, JSON, etc.)
4. **Visualization**: Integrate with visualization libraries for graphical output

## Limitations and Considerations

1. **Website Changes**: The scraper may break if Cricinfo changes their HTML structure
2. **Access Restrictions**: Some pages may be restricted or require authentication
3. **Data Completeness**: Not all statistics may be available or accessible
4. **Performance**: Complex queries and calculations may be time-consuming
5. **Legal Considerations**: Respect Cricinfo's terms of service and rate limits

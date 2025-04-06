"""
Statsguru Data Extraction Module

This module provides functionality to extract and parse data from Cricinfo's Statsguru HTML responses.
It handles HTML parsing, table extraction, and data cleaning.
"""

import pandas as pd
from bs4 import BeautifulSoup
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StatguruDataExtractor:
    """
    Extracts and parses data from Cricinfo's Statsguru HTML responses.
    """
    
    def __init__(self):
        """Initialize the data extractor."""
        pass
    
    def extract_tables(self, html_content):
        """
        Extract tables from HTML content
        
        Args:
            html_content (str): HTML content from Statsguru
            
        Returns:
            list: List of BeautifulSoup table elements
        """
        if not html_content:
            logger.error("No HTML content provided for extraction")
            return []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = soup.find_all('table', class_='engineTable')
            
            if not tables:
                logger.warning("No tables found in HTML content")
            else:
                logger.info(f"Found {len(tables)} tables in HTML content")
            
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            return []
    
    def parse_table(self, table):
        """
        Parse HTML table into pandas DataFrame
        
        Args:
            table (BeautifulSoup element): HTML table element
            
        Returns:
            pandas.DataFrame: Parsed table data
        """
        if not table:
            logger.error("No table provided for parsing")
            return None
        
        try:
            # Extract headers
            headers = []
            header_row = table.find('thead')
            
            if header_row:
                header_cells = header_row.find_all('th')
                for th in header_cells:
                    # Clean header text
                    header = th.text.strip()
                    headers.append(header)
            else:
                # Try to find headers in the first row if thead is not present
                first_row = table.find('tr')
                if first_row:
                    header_cells = first_row.find_all(['th', 'td'])
                    for cell in header_cells:
                        header = cell.text.strip()
                        headers.append(header)
            
            # Extract data rows
            rows = []
            body = table.find('tbody') or table
            data_rows = body.find_all('tr')
            
            for tr in data_rows:
                # Skip header row if we're using tbody
                if tr.find('th') and body == table:
                    continue
                
                row = []
                cells = tr.find_all(['td', 'th'])
                for cell in cells:
                    # Clean cell text
                    cell_text = cell.text.strip()
                    row.append(cell_text)
                
                # Only add rows with data
                if any(cell for cell in row):
                    rows.append(row)
            
            # Create DataFrame
            if headers and rows:
                # Ensure rows match header length
                rows = [row[:len(headers)] for row in rows]
                # Pad short rows
                for row in rows:
                    while len(row) < len(headers):
                        row.append(None)
                
                df = pd.DataFrame(rows, columns=headers)
                logger.info(f"Successfully parsed table with {len(df)} rows and {len(df.columns)} columns")
                return df
            else:
                logger.warning("Failed to extract headers or rows from table")
                return None
        except Exception as e:
            logger.error(f"Error parsing table: {e}")
            return None
    
    def clean_data(self, df):
        """
        Clean and preprocess the data
        
        Args:
            df (pandas.DataFrame): Raw data frame
            
        Returns:
            pandas.DataFrame: Cleaned data frame
        """
        if df is None or df.empty:
            logger.warning("No data to clean")
            return df
        
        try:
            # Make a copy to avoid modifying the original
            df_clean = df.copy()
            
            # Remove any completely empty rows or columns
            df_clean = df_clean.dropna(how='all').dropna(axis=1, how='all')
            
            # Handle common column names and formats in Statsguru
            
            # Rename columns to standardized names
            column_mapping = {
                'Runs': 'Runs',
                'Wkts': 'Wickets',
                'Balls': 'Balls',
                'Mins': 'Minutes',
                'BF': 'Balls_Faced',
                'SR': 'Strike_Rate',
                'Econ': 'Economy',
                'Ave': 'Average',
                'Mat': 'Matches',
                'Inns': 'Innings',
                '100': 'Hundreds',
                '50': 'Fifties',
                '4s': 'Fours',
                '6s': 'Sixes',
                'Ct': 'Catches',
                'St': 'Stumpings',
                'Player': 'Player',
                'Opposition': 'Opposition',
                'Ground': 'Ground',
                'Start Date': 'Date',
                'Scorecard': 'Scorecard'
            }
            
            # Apply column renaming where matches exist
            for old_col, new_col in column_mapping.items():
                if old_col in df_clean.columns:
                    df_clean = df_clean.rename(columns={old_col: new_col})
            
            # Convert numeric columns
            numeric_columns = [
                'Runs', 'Wickets', 'Balls', 'Minutes', 'Balls_Faced', 
                'Strike_Rate', 'Economy', 'Average', 'Matches', 'Innings',
                'Hundreds', 'Fifties', 'Fours', 'Sixes', 'Catches', 'Stumpings'
            ]
            
            for col in numeric_columns:
                if col in df_clean.columns:
                    # Handle special cases like '-' or 'DNB'
                    df_clean[col] = df_clean[col].replace(['-', 'DNB', 'TDNB', 'absent', 'not out'], None)
                    
                    # Convert to numeric, handling errors
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Handle date columns
            date_columns = ['Date']
            for col in date_columns:
                if col in df_clean.columns:
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    except:
                        logger.warning(f"Failed to convert {col} to datetime")
            
            # Remove any rows that are just headers repeated in the data
            if 'Player' in df_clean.columns:
                df_clean = df_clean[~df_clean['Player'].str.contains('Player', na=False)]
            
            logger.info(f"Data cleaning complete. Shape: {df_clean.shape}")
            return df_clean
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return df
    
    def extract_player_ids(self, html_content):
        """
        Extract player IDs from HTML content
        
        Args:
            html_content (str): HTML content from Statsguru
            
        Returns:
            dict: Dictionary mapping player names to IDs
        """
        if not html_content:
            return {}
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            player_links = soup.find_all('a', href=re.compile(r'/ci/content/player/\d+\.html'))
            
            player_ids = {}
            for link in player_links:
                player_name = link.text.strip()
                match = re.search(r'/ci/content/player/(\d+)\.html', link['href'])
                if match and player_name:
                    player_id = match.group(1)
                    player_ids[player_name] = player_id
            
            logger.info(f"Extracted {len(player_ids)} player IDs")
            return player_ids
        except Exception as e:
            logger.error(f"Error extracting player IDs: {e}")
            return {}
    
    def extract_data_from_html(self, html_content):
        """
        Extract all relevant data from HTML content
        
        Args:
            html_content (str): HTML content from Statsguru
            
        Returns:
            dict: Dictionary containing extracted data
        """
        if not html_content:
            logger.error("No HTML content provided")
            return {"tables": [], "player_ids": {}}
        
        tables = self.extract_tables(html_content)
        parsed_tables = []
        
        for table in tables:
            df = self.parse_table(table)
            if df is not None and not df.empty:
                df = self.clean_data(df)
                parsed_tables.append(df)
        
        player_ids = self.extract_player_ids(html_content)
        
        return {
            "tables": parsed_tables,
            "player_ids": player_ids
        }


# Example usage
if __name__ == "__main__":
    from statsguru_query import StatguruQuery
    
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
    
    # Print results
    print(f"Extracted {len(data['tables'])} tables")
    for i, table in enumerate(data['tables']):
        print(f"\nTable {i+1}:")
        print(table.head())
    
    print("\nPlayer IDs:")
    for name, player_id in data['player_ids'].items():
        print(f"{name}: {player_id}")
    
    query.close()

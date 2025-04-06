# Cricinfo Statsguru AI Agent Documentation

## Overview

The Cricinfo Statsguru AI Agent is a powerful tool that allows users to extract cricket statistics data from Cricinfo's Statsguru, execute queries, and perform calculations on the extracted data. The agent provides a natural language interface for interacting with Statsguru, making it easy to access and analyze cricket statistics for various purposes, including creating YouTube videos comparing cricket players.

## Features

- **Natural Language Interface**: Interact with the agent using simple English commands
- **Data Extraction**: Extract cricket statistics data from Cricinfo's Statsguru
- **Player Comparisons**: Compare statistics between players (e.g., Sachin Tendulkar vs. Virat Kohli)
- **Advanced Calculations**: Calculate various cricket statistics metrics including:
  - Batting impact scores
  - Era-adjusted statistics
  - Weighted performance metrics
  - Match-winning contributions
  - Batting indices
- **Visualizations**: Generate visualizations for statistical comparisons
- **Report Generation**: Create comprehensive comparison reports

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages:
  - requests
  - beautifulsoup4
  - pandas
  - matplotlib
  - seaborn
  - numpy

### Setup

1. Clone the repository:
```
git clone https://github.com/yourusername/cricinfo-agent.git
cd cricinfo-agent
```

2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The agent can be used through a command-line interface provided by `main.py`. There are two modes of operation:

1. **Interactive Mode**:
```
python main.py --interactive
```

2. **Single Command Mode**:
```
python main.py --command "Get batting statistics for Sachin Tendulkar in test cricket"
```

### Example Commands

#### Query Commands
- "Get batting statistics for Sachin Tendulkar in test cricket"
- "Show bowling figures for Dale Steyn against India"
- "Find batting stats for Virat Kohli in ODIs from 2015 to 2020"

#### Comparison Commands
- "Compare Sachin Tendulkar and Virat Kohli in test cricket"
- "Compare bowling stats of James Anderson and Dale Steyn"

#### Calculation Commands
- "Calculate batting impact score for Virat Kohli in test cricket"
- "Compute era-adjusted stats for Don Bradman"
- "Determine match-winning contributions for Ricky Ponting"

#### Visualization Commands
- "Visualize career progression for Sachin Tendulkar"
- "Plot batting stats by opposition for Virat Kohli"
- "Create comparison chart for Sachin Tendulkar and Virat Kohli"

#### Report Commands
- "Generate report for Sachin Tendulkar and Virat Kohli in test cricket"
- "Create comprehensive comparison report for Ricky Ponting and Steve Smith"

## Architecture

The agent is built with a modular architecture consisting of the following components:

1. **Query Interface** (`statsguru_query.py`): Constructs and executes queries to Cricinfo's Statsguru
2. **Data Extractor** (`statsguru_extractor.py`): Extracts and parses data from HTML responses
3. **Data Processor** (`statsguru_processor.py`): Processes and analyzes cricket statistics data
4. **Cricket Statistics Calculator** (`cricket_stats_calculator.py`): Implements specialized cricket statistics calculations
5. **User Interaction Module** (`user_interaction.py`): Provides a natural language interface for users

## Module Details

### Query Interface

The query interface handles the construction and execution of queries to Cricinfo's Statsguru. It provides classes for building query URLs with various parameters and executing HTTP requests.

Key classes:
- `StatguruQueryBuilder`: Builds query URLs based on user parameters
- `StatguruQueryExecutor`: Executes HTTP requests and handles responses
- `StatguruQuery`: Main interface for constructing and executing queries

### Data Extractor

The data extractor handles the extraction and parsing of data from HTML responses. It uses BeautifulSoup to parse HTML and extract tabular data.

Key classes:
- `StatguruDataExtractor`: Extracts and parses data from HTML responses

### Data Processor

The data processor provides functions for processing and analyzing cricket statistics data. It includes functions for calculating various cricket statistics metrics.

Key classes:
- `StatguruDataProcessor`: Processes and analyzes cricket statistics data

### Cricket Statistics Calculator

The cricket statistics calculator implements specialized cricket statistics calculations for player comparisons and analysis. It builds on the data processor to provide more advanced calculations.

Key classes:
- `CricketStatsCalculator`: Implements specialized cricket statistics calculations

### User Interaction Module

The user interaction module provides a natural language interface for users to interact with the agent. It parses user commands and executes the appropriate actions.

Key classes:
- `CricinfoAgent`: Main agent class that handles user interactions and executes commands

## Creating YouTube Videos with the Agent

The agent is particularly useful for creating YouTube videos comparing cricket players like Sachin Tendulkar and Virat Kohli. Here's a workflow for using the agent to create such videos:

1. **Extract Player Data**:
   ```
   python main.py --command "Get batting statistics for Sachin Tendulkar in test cricket"
   python main.py --command "Get batting statistics for Virat Kohli in test cricket"
   ```

2. **Generate Comparisons and Visualizations**:
   ```
   python main.py --command "Compare Sachin Tendulkar and Virat Kohli in test cricket"
   python main.py --command "Visualize career progression for Sachin Tendulkar"
   python main.py --command "Visualize career progression for Virat Kohli"
   ```

3. **Generate Comprehensive Report**:
   ```
   python main.py --command "Generate report for Sachin Tendulkar and Virat Kohli in test cricket"
   ```

4. **Use the Generated Visualizations and Report in Your Video**:
   - The visualizations are saved in the `output` directory
   - The report is saved as a Markdown file in the `output` directory
   - Use these assets in your video editing software

## Limitations and Considerations

- The agent relies on the structure of Cricinfo's Statsguru, which may change over time
- Some pages may be restricted or require authentication
- The agent may be subject to rate limiting by Cricinfo
- Not all statistics may be available or accessible
- The agent respects Cricinfo's terms of service and rate limits

## Troubleshooting

### Common Issues

1. **No Data Found**:
   - Check that the player name is spelled correctly
   - Try using a different format (test, odi, t20i)
   - Check if the player has played in the specified format

2. **Error Executing Query**:
   - Check your internet connection
   - Cricinfo may be experiencing issues or may have changed their structure
   - You may be rate limited; try again later

3. **Visualization Errors**:
   - Check that the required data is available
   - Some visualizations require specific data (e.g., dates for career progression)

### Getting Help

If you encounter issues not covered in this documentation, you can:
- Use the help command: `python main.py --command "help"`
- Check the GitHub repository for updates and issues
- Contact the developer for support

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ESPNCricinfo for providing the Statsguru database
- The open-source community for the libraries used in this project

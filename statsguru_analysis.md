# Cricinfo Statsguru Structure Analysis

## Overview
Cricinfo's Statsguru is a comprehensive cricket statistics database that allows users to query and analyze cricket data across different formats, players, teams, and time periods. The interface provides powerful filtering capabilities and various output formats.

## Key Components

### 1. Cricket Formats
- Tests
- ODIs (One Day Internationals)
- T20Is (Twenty20 Internationals)
- All Test/ODI/T20I (combined)
- Twenty20 (domestic)
- Women's T20Is
- Other formats (Women's Tests, Women's ODIs, etc.)

### 2. Analysis Types
- Batting
- Bowling
- Fielding
- All-round
- Partnership
- Team
- Umpire and referee
- Aggregate/overall

### 3. Query Parameters
- Team selection
- Opposition selection
- Venue type (home, away, neutral)
- Host country
- Ground
- Date range
- Season
- Match result filters
- Output format options (overall figures, innings lists, match totals, etc.)

### 4. URL Structure
The Statsguru interface uses URL parameters to construct queries. The base URL is:
```
https://stats.espncricinfo.com/ci/engine/stats/index.html
```

Parameters are appended to this URL in the format:
```
?param1=value1;param2=value2;...
```

Common parameters include:
- `class`: Cricket format (1 for Tests, 2 for ODIs, 3 for T20Is)
- `type`: Analysis type (batting, bowling, etc.)
- `template`: Output format
- Various filter parameters for teams, opposition, venues, etc.

### 5. Player Analysis
The Statsguru analysis section allows users to search for specific players by entering their names. This functionality is crucial for player comparisons.

### 6. Data Presentation
Results are presented in tabular format with sortable columns. The data includes various statistics depending on the query type (batting averages, bowling figures, etc.).

## Access Limitations
- Direct access to some player pages and comparison interfaces may be restricted
- Web scraping may require handling of access restrictions and rate limiting

## Extraction Strategy
1. Construct appropriate query URLs based on user requirements
2. Use web scraping to extract tabular data from query results
3. Parse and process the extracted data for analysis and visualization

## Demo Resources
Statsguru provides demo pages that illustrate how to construct queries for batsmen and bowlers, which will be valuable references for implementing the query interface.

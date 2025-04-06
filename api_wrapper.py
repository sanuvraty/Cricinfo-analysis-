"""
API wrapper for Cricinfo Statsguru AI Agent

This module provides a Flask API wrapper around the Cricinfo Statsguru AI Agent,
allowing it to be accessed via HTTP requests from a web application.
"""

from flask import Flask, request, jsonify
import sys
import os
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the agent
from user_interaction import CricinfoAgent

# Initialize Flask app
app = Flask(__name__)

# Initialize agent
agent = CricinfoAgent()

@app.route('/api/command', methods=['POST'])
def process_command():
    """
    Process a command sent to the agent
    
    Request body:
    {
        "command": "Command text"
    }
    
    Returns:
    JSON response with command results
    """
    try:
        data = request.json
        
        if not data or 'command' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing command parameter'
            }), 400
        
        command = data['command']
        logger.info(f"Processing command: {command}")
        
        # Process command with agent
        result = agent.process_command(command)
        
        # Convert any non-serializable objects to strings
        result = make_json_serializable(result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing command: {e}")
        return jsonify({
            'success': False,
            'error': f"Error processing command: {str(e)}"
        }), 500

@app.route('/api/help', methods=['GET'])
def get_help():
    """
    Get help information about available commands
    
    Returns:
    JSON response with help information
    """
    try:
        result = agent.process_command("help")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error getting help: {e}")
        return jsonify({
            'success': False,
            'error': f"Error getting help: {str(e)}"
        }), 500

@app.route('/api/query', methods=['POST'])
def execute_query():
    """
    Execute a query to Statsguru
    
    Request body:
    {
        "player": "Player name",
        "format": "test|odi|t20i",
        "analysis_type": "batting|bowling"
    }
    
    Returns:
    JSON response with query results
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Missing query parameters'
            }), 400
        
        # Construct command from parameters
        player = data.get('player', '')
        format_type = data.get('format', 'test')
        analysis_type = data.get('analysis_type', 'batting')
        
        command = f"Get {analysis_type} statistics for {player} in {format_type} cricket"
        logger.info(f"Executing query command: {command}")
        
        # Process command with agent
        result = agent.process_command(command)
        
        # Convert any non-serializable objects to strings
        result = make_json_serializable(result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return jsonify({
            'success': False,
            'error': f"Error executing query: {str(e)}"
        }), 500

@app.route('/api/compare', methods=['POST'])
def compare_players():
    """
    Compare statistics between two players
    
    Request body:
    {
        "player1": "First player name",
        "player2": "Second player name",
        "format": "test|odi|t20i",
        "analysis_type": "batting|bowling"
    }
    
    Returns:
    JSON response with comparison results
    """
    try:
        data = request.json
        
        if not data or 'player1' not in data or 'player2' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing player parameters'
            }), 400
        
        # Construct command from parameters
        player1 = data.get('player1', '')
        player2 = data.get('player2', '')
        format_type = data.get('format', 'test')
        analysis_type = data.get('analysis_type', 'batting')
        
        command = f"Compare {player1} and {player2} in {format_type} cricket"
        if analysis_type == 'bowling':
            command = f"Compare bowling stats of {player1} and {player2} in {format_type} cricket"
        
        logger.info(f"Executing comparison command: {command}")
        
        # Process command with agent
        result = agent.process_command(command)
        
        # Convert any non-serializable objects to strings
        result = make_json_serializable(result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error comparing players: {e}")
        return jsonify({
            'success': False,
            'error': f"Error comparing players: {str(e)}"
        }), 500

@app.route('/api/visualize', methods=['POST'])
def create_visualization():
    """
    Create visualizations for player statistics
    
    Request body:
    {
        "player1": "First player name",
        "player2": "Second player name (optional)",
        "format": "test|odi|t20i",
        "visualization": "basic|career_progression|by_opposition|comparison"
    }
    
    Returns:
    JSON response with visualization results
    """
    try:
        data = request.json
        
        if not data or 'player1' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing player parameter'
            }), 400
        
        # Construct command from parameters
        player1 = data.get('player1', '')
        player2 = data.get('player2', '')
        format_type = data.get('format', 'test')
        visualization = data.get('visualization', 'basic')
        
        if player2:
            command = f"Visualize {visualization} for {player1} and {player2} in {format_type} cricket"
        else:
            command = f"Visualize {visualization} for {player1} in {format_type} cricket"
        
        logger.info(f"Executing visualization command: {command}")
        
        # Process command with agent
        result = agent.process_command(command)
        
        # Convert any non-serializable objects to strings
        result = make_json_serializable(result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return jsonify({
            'success': False,
            'error': f"Error creating visualization: {str(e)}"
        }), 500

@app.route('/api/report', methods=['POST'])
def generate_report():
    """
    Generate a comprehensive comparison report
    
    Request body:
    {
        "player1": "First player name",
        "player2": "Second player name",
        "format": "test|odi|t20i"
    }
    
    Returns:
    JSON response with report results
    """
    try:
        data = request.json
        
        if not data or 'player1' not in data or 'player2' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing player parameters'
            }), 400
        
        # Construct command from parameters
        player1 = data.get('player1', '')
        player2 = data.get('player2', '')
        format_type = data.get('format', 'test')
        
        command = f"Generate report for {player1} and {player2} in {format_type} cricket"
        logger.info(f"Executing report command: {command}")
        
        # Process command with agent
        result = agent.process_command(command)
        
        # Convert any non-serializable objects to strings
        result = make_json_serializable(result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return jsonify({
            'success': False,
            'error': f"Error generating report: {str(e)}"
        }), 500

@app.route('/api/calculate', methods=['POST'])
def calculate_stats():
    """
    Calculate statistics for a player
    
    Request body:
    {
        "player": "Player name",
        "format": "test|odi|t20i",
        "calculation": "impact|era_adjusted|weighted|match_winning|batting_index"
    }
    
    Returns:
    JSON response with calculation results
    """
    try:
        data = request.json
        
        if not data or 'player' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing player parameter'
            }), 400
        
        # Construct command from parameters
        player = data.get('player', '')
        format_type = data.get('format', 'test')
        calculation = data.get('calculation', 'impact')
        
        command = f"Calculate {calculation} for {player} in {format_type} cricket"
        logger.info(f"Executing calculation command: {command}")
        
        # Process command with agent
        result = agent.process_command(command)
        
        # Convert any non-serializable objects to strings
        result = make_json_serializable(result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        return jsonify({
            'success': False,
            'error': f"Error calculating stats: {str(e)}"
        }), 500

@app.route('/api/output/<path:filename>', methods=['GET'])
def get_output_file(filename):
    """
    Get an output file (visualization, report, etc.)
    
    Args:
        filename: Name of the file to retrieve
        
    Returns:
        File content
    """
    try:
        output_dir = os.path.join(os.getcwd(), 'output')
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': f"File not found: {filename}"
            }), 404
        
        # Determine file type
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            with open(file_path, 'rb') as f:
                return f.read(), 200, {'Content-Type': 'image/png' if filename.endswith('.png') else 'image/jpeg'}
        elif filename.endswith('.md'):
            with open(file_path, 'r') as f:
                content = f.read()
                return jsonify({
                    'success': True,
                    'content': content
                })
        else:
            with open(file_path, 'r') as f:
                content = f.read()
                return jsonify({
                    'success': True,
                    'content': content
                })
    
    except Exception as e:
        logger.error(f"Error retrieving file: {e}")
        return jsonify({
            'success': False,
            'error': f"Error retrieving file: {str(e)}"
        }), 500

def make_json_serializable(obj):
    """
    Convert any non-JSON-serializable objects to strings
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

@app.after_request
def add_cors_headers(response):
    """Add CORS headers to allow cross-origin requests"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    
    Returns:
    JSON response with health status
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Cricinfo Statsguru AI Agent API is running'
    })

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port)

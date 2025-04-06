"""
Main script for running the Cricinfo Statsguru AI Agent

This script provides a simple command-line interface for interacting with
the Cricinfo Statsguru AI Agent.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the agent
from user_interaction import CricinfoAgent

def main():
    """Main function to run the Cricinfo Statsguru AI Agent."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cricinfo Statsguru AI Agent')
    parser.add_argument('--command', type=str, help='Command to execute')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    args = parser.parse_args()
    
    # Initialize agent
    agent = CricinfoAgent()
    
    try:
        if args.interactive:
            # Run in interactive mode
            print("Cricinfo Statsguru AI Agent")
            print("Type 'exit' or 'quit' to exit")
            print("Type 'help' for available commands")
            
            while True:
                # Get command from user
                command = input("\nEnter command: ")
                
                # Check if user wants to exit
                if command.lower() in ['exit', 'quit']:
                    break
                
                # Process command
                result = agent.process_command(command)
                
                # Display result
                if result['success']:
                    print("\nCommand executed successfully")
                    
                    # Display specific results based on command type
                    if 'tables' in result:
                        print(f"\nFound {len(result['tables'])} tables")
                        if result['tables'] and len(result['tables']) > 0:
                            print(f"\nFirst table has {len(result['tables'][0])} records")
                    
                    if 'comparison' in result:
                        print("\nComparison completed")
                        if 'visualizations' in result:
                            print(f"\nVisualizations saved to: {', '.join(result['visualizations'].values())}")
                    
                    if 'results' in result:
                        print("\nResults:")
                        for key, value in result['results'].items():
                            print(f"  {key}: {value}")
                    
                    if 'visualization_path' in result:
                        print(f"\nVisualization saved to: {result['visualization_path']}")
                    
                    if 'report_path' in result:
                        print(f"\nReport saved to: {result['report_path']}")
                        
                    if 'help' in result:
                        print(result['help'])
                else:
                    print(f"\nError: {result.get('error', 'Unknown error')}")
        
        elif args.command:
            # Execute single command
            result = agent.process_command(args.command)
            
            # Display result
            if result['success']:
                print("Command executed successfully")
                
                # Display specific results based on command type
                if 'tables' in result:
                    print(f"Found {len(result['tables'])} tables")
                
                if 'comparison' in result:
                    print("Comparison completed")
                    if 'visualizations' in result:
                        print(f"Visualizations saved to: {', '.join(result['visualizations'].values())}")
                
                if 'results' in result:
                    print("Results:")
                    for key, value in result['results'].items():
                        print(f"  {key}: {value}")
                
                if 'visualization_path' in result:
                    print(f"Visualization saved to: {result['visualization_path']}")
                
                if 'report_path' in result:
                    print(f"Report saved to: {result['report_path']}")
                    
                if 'help' in result:
                    print(result['help'])
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        else:
            # No arguments provided, show help
            print("Cricinfo Statsguru AI Agent")
            print("Run with --interactive for interactive mode")
            print("Run with --command 'your command' to execute a single command")
            print("Example: python main.py --command 'Get batting statistics for Sachin Tendulkar in test cricket'")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Close agent
        agent.close()

if __name__ == "__main__":
    main()

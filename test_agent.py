"""
Test script for Cricinfo Statsguru AI Agent

This script tests the functionality of the Cricinfo Statsguru AI Agent
by executing various commands and verifying the results.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from user_interaction import CricinfoAgent

def test_agent():
    """Test the functionality of the Cricinfo Statsguru AI Agent."""
    logger.info("Starting agent tests")
    
    # Create output directory for test results
    output_dir = os.path.join(os.getcwd(), 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize agent
    agent = CricinfoAgent()
    
    # Define test cases
    test_cases = [
        {
            "name": "Help Command",
            "command": "help",
            "expected_success": True
        },
        {
            "name": "Basic Query",
            "command": "Get batting statistics for Sachin Tendulkar in test cricket",
            "expected_success": True
        },
        {
            "name": "Player Comparison",
            "command": "Compare Sachin Tendulkar and Virat Kohli in test cricket",
            "expected_success": True
        },
        {
            "name": "Impact Score Calculation",
            "command": "Calculate batting impact score for Virat Kohli in test cricket",
            "expected_success": True
        },
        {
            "name": "Career Progression Visualization",
            "command": "Visualize career progression for Sachin Tendulkar",
            "expected_success": True
        },
        {
            "name": "Comparison Report Generation",
            "command": "Generate report for Sachin Tendulkar and Virat Kohli in test cricket",
            "expected_success": True
        },
        {
            "name": "Bowling Statistics Query",
            "command": "Get bowling statistics for James Anderson in test cricket",
            "expected_success": True
        },
        {
            "name": "Invalid Player Query",
            "command": "Get batting statistics for NonExistentPlayer in test cricket",
            "expected_success": False
        }
    ]
    
    # Run tests
    results = []
    for i, test in enumerate(test_cases):
        logger.info(f"Running test {i+1}/{len(test_cases)}: {test['name']}")
        
        try:
            # Execute command
            result = agent.process_command(test["command"])
            
            # Check if success matches expected
            success_match = result["success"] == test["expected_success"]
            
            # Save result
            test_result = {
                "test_name": test["name"],
                "command": test["command"],
                "expected_success": test["expected_success"],
                "actual_success": result["success"],
                "success_match": success_match,
                "result": result
            }
            
            # Log result
            if success_match:
                logger.info(f"Test {i+1} PASSED: {test['name']}")
            else:
                logger.error(f"Test {i+1} FAILED: {test['name']}")
                logger.error(f"Expected success: {test['expected_success']}, Actual: {result['success']}")
                if not result["success"]:
                    logger.error(f"Error: {result.get('error', 'Unknown error')}")
            
            results.append(test_result)
            
        except Exception as e:
            logger.error(f"Test {i+1} ERROR: {test['name']}")
            logger.error(f"Exception: {str(e)}")
            
            results.append({
                "test_name": test["name"],
                "command": test["command"],
                "expected_success": test["expected_success"],
                "actual_success": False,
                "success_match": False,
                "error": str(e)
            })
    
    # Save test results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate test summary
    passed = sum(1 for r in results if r["success_match"])
    total = len(results)
    
    summary = f"""
    # Cricinfo Statsguru AI Agent Test Results
    
    - Total Tests: {total}
    - Passed: {passed}
    - Failed: {total - passed}
    - Success Rate: {passed/total*100:.2f}%
    
    ## Test Details
    
    | # | Test Name | Command | Expected | Actual | Result |
    |---|-----------|---------|----------|--------|--------|
    """
    
    for i, r in enumerate(results):
        result_str = "PASS" if r["success_match"] else "FAIL"
        summary += f"| {i+1} | {r['test_name']} | `{r['command']}` | {r['expected_success']} | {r.get('actual_success', 'ERROR')} | {result_str} |\n"
    
    with open(os.path.join(output_dir, 'test_summary.md'), 'w') as f:
        f.write(summary)
    
    logger.info(f"Tests completed: {passed}/{total} passed ({passed/total*100:.2f}%)")
    
    # Close agent
    agent.close()
    
    return results

if __name__ == "__main__":
    test_agent()

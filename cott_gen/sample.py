#!/usr/bin/env python3
"""Error Analysis and Sampling Tool for Ego-R1 QA Results - categorizes errors in processed results."""

import os
import json
import argparse
from tqdm import tqdm
from utils import llm_fuzzy_match


def load_json(file_path: str) -> dict:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
        

def check_json(json_data: dict, explicit_answer: bool = False) -> bool:
    """Check if model's answer matches ground truth."""
    assert isinstance(json_data, dict), "Input must be a dictionary"
    
    if explicit_answer:
        # Use LLM fuzzy matching for flexible answer comparison
        answer = llm_fuzzy_match(
            json_data['question'], 
            json_data['options'], 
            json_data['cot'][-1]["answer"]
        )
        answer = answer.strip("```").strip("```")  # Clean markdown formatting
    else:
        answer = json_data['cot'][-1]["answer"]  # Direct string comparison

    return answer == json_data['ground_truth']


def check_json_error(json_data: dict) -> bool:
    """Check if any CoT step contains an error in observations."""
    assert isinstance(json_data, dict), "Input must be a dictionary"
    
    for step in json_data.get('cot', []):  # Check each step for errors
        if 'observation' in step:
            obs_str = str(step['observation']).lower()
            if 'error' in obs_str:
                return True
    return False


def check_json_no_answer(json_data: dict) -> bool:
    """Check if final step is missing an answer."""
    assert isinstance(json_data, dict), "Input must be a dictionary"
    
    if 'cot' in json_data and len(json_data['cot']) > 0:
        last_step = json_data['cot'][-1]
        if 'answer' not in last_step:  # Final step should contain an answer
            return True
    return False


def check_json_na(json_data: dict) -> bool:
    """Check if final answer is N/A."""
    assert isinstance(json_data, dict), "Input must be a dictionary"
    
    if 'cot' in json_data and len(json_data['cot']) > 0:
        last_step = json_data['cot'][-1]
        if 'answer' in last_step:
            answer = str(last_step['answer']).lower()
            return answer in ['n/a', 'na']  # Check for N/A formats
    return False


def check_json_tool_mismatch(json_data: dict) -> bool:
    """Check mismatch between tool mentioned in thought and tool actually called."""
    assert isinstance(json_data, dict), "Input must be a dictionary"
    
    tool_names = ['video_llm', 'vlm', 'rag']
    
    for step in json_data.get('cot', []):  # Analyze each CoT step
        if 'thought' not in step:  # Skip steps without thought
            continue
            
        thought = step['thought'].lower()
        
        # Get last meaningful sentence from thought
        sentences = thought.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            continue
            
        last_sentence = sentences[-1]
        
        # Check if any tool is mentioned in last sentence
        last_mentioned_tool = None
        for tool in tool_names:
            if f'`{tool}`' in last_sentence or f' {tool} ' in last_sentence:
                last_mentioned_tool = tool.lower()
                break
        
        # Mismatch if tool mentioned but not called
        if last_mentioned_tool and 'tool' not in step:
            return True
            
        # Mismatch if different tool called than mentioned
        if last_mentioned_tool and 'tool' in step:
            called_tool = step['tool'].get('name', '').lower()
            if last_mentioned_tool != called_tool:
                return True
    
    return False


def check_A3_rag_error(json_data: dict) -> bool:
    """Check for A3-specific RAG URL errors."""
    assert isinstance(json_data, dict), "Input must be a dictionary"
    
    if 'cot' in json_data and len(json_data['cot']) > 0:
        for step in json_data['cot']:
            if 'observation' in step: 
                obs_str = str(step['observation'])
                if "Error: 'RAG_URL" in obs_str:  # A3-specific error pattern
                    return True
    return False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Ego-R1 QA results and categorize errors"
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='results_20250420_232714_aobs',
        help='Directory containing result JSON files to analyze'
    )
    parser.add_argument(
        '--sample_dir', 
        type=str, 
        default='errors/',
        help='Directory to save error analysis results'
    )
    parser.add_argument(
        '--explicit_answer', 
        action='store_true',
        help='Use LLM fuzzy matching for answer comparison'
    )
    parser.add_argument(
        "--sample_type", 
        type=str, 
        default="correct", 
        choices=["correct", "error", "na", "tool_mismatch", "no_answer"],
        help='Type of samples to extract'
    )
    parser.add_argument(
        "--identity", 
        type=str, 
        default="A1",
        help='Identity being analyzed'
    )

    return parser.parse_args()


def main():
    """Analyze results and generate error categorization reports."""
    args = parse_arguments()
    data_dir = args.data_dir
    
    # Create output directory
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    
    # Initialize counters and error lists
    matched = 0
    total = 0
    
    error_list_error = []           # General errors in observations
    error_list_na = []              # N/A answers
    error_list_tool_mismatch = []   # Thought-action tool mismatches
    error_list_no_answer = []       # Missing answers
    error_list_A3_error = []        # A3-specific RAG errors
    
    error_types = ["error", "na", "tool_mismatch", "no_answer"]
    
    # Process each file in data directory
    for file in tqdm(os.listdir(data_dir), desc="Analyzing files"):
        dp_path = os.path.join(data_dir, file)
        
        if os.path.isfile(dp_path):  # Only process files, skip directories
            try:
                json_data = load_json(dp_path)
                total += 1
                
                # Check A3-specific errors
                if args.identity == "A3":
                    if check_A3_rag_error(json_data):
                        error_list_A3_error.append(json_data)
                
                # Categorize by error type
                for error_type in error_types:
                    if error_type == "error":
                        if check_json_error(json_data):
                            error_list_error.append(json_data)
                    elif error_type == "na":
                        if check_json_na(json_data):
                            error_list_na.append(json_data)
                    elif error_type == "tool_mismatch":
                        if check_json_tool_mismatch(json_data):
                            error_list_tool_mismatch.append(json_data)
                    elif error_type == "no_answer":
                        if check_json_no_answer(json_data):
                            error_list_no_answer.append(json_data)
                            
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue
    
    # Save error lists to JSON files
    error_files = {
        f"error_list_error_{args.identity}.json": error_list_error,
        f"error_list_na_{args.identity}.json": error_list_na,
        f"error_list_no_answer_{args.identity}.json": error_list_no_answer,
        f"error_list_tool_mismatch_{args.identity}.json": error_list_tool_mismatch,
    }
    
    # Save all error categories
    for filename, error_list in error_files.items():
        filepath = os.path.join(args.sample_dir, filename)
        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(error_list, f, indent=4)
        print(f"Saved {len(error_list)} items to {filename}")
    
    # Save A3-specific errors if applicable
    if args.identity == "A3":   
        a3_filepath = os.path.join(args.sample_dir, f"error_list_A3_error_{args.identity}.json")
        with open(a3_filepath, "w", encoding='utf-8') as f:
            json.dump(error_list_A3_error, f, indent=4)
        print(f"Saved {len(error_list_A3_error)} A3-specific errors")
    
    # Print summary statistics
    print(f"\nAnalysis Summary:")
    print(f"Total files processed: {total}")
    print(f"General errors: {len(error_list_error)}")
    print(f"N/A answers: {len(error_list_na)}")
    print(f"Tool mismatches: {len(error_list_tool_mismatch)}")
    print(f"No answers: {len(error_list_no_answer)}")
    if args.identity == "A3":
        print(f"A3 RAG errors: {len(error_list_A3_error)}")


if __name__ == '__main__':
    main()

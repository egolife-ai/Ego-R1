"""
Data processing module for EgoLife dataset.

This module handles post-processing of training data, including format conversion,
data validation, and preparation for different training stages (SFT/RL).
"""

import ast
import argparse
import datetime
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chardet
import pandas as pd
import transformers
from datasets import concatenate_datasets, load_dataset


# Constants
IDENTITY_MAPPING = {
    "A1": "JAKE",
    "A2": "ALICE",
    "A3": "TASHA",
    "A4": "LUCIA",
    "A5": "KATRINA",
    "A6": "SHURE"
}

IDENTITIES = ["A1", "A2", "A3", "A4", "A5", "A6"]
MAX_COT_LENGTH = 8
EXPECTED_BENCHMARK_ITEMS = 25

# Template configurations
PROMPT_TEMPLATE = """
## INSTRUCTIONS
Answer the given question. You must conduct reasoning inside <think> and </think> first every time before you get new information. \
After reasoning, if you find you lack some knowledge, you can call a tool from [rag, video_llm, vlm] by \
<tool> query </tool> and it will return the information between <information> and </information>. \
You can use tools as many times as your want. If you find no further external knowledge needed, \
you can provide the answer inside <answer> and </answer> after another thinking.

The tools you can use are:
{
    "name": "rag",
    "description": "Use this tool to search for information in the RAG database.",
    "arguments": {
        "type": "object",
        "properties": {
            "level": {
                "type": "str",
                "description": "The granularity of the search, choose from week|day|hour"
            },
            "keywords": {
                "type": "List[str]",
                "description": "The keywords to search for in the RAG database."
            },
            "start_time": {
                "type": "str",
                "description": "The timestamp of the start time of the search. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))."
            },
            "query_time": {
                "type": "str",
                "description": "The timestamp of the query that was proposed by the user."
            }
        },
        "required": ["level", "keywords", "start_time", "query_time"]
    }
}
{
    "name": "video_llm",
    "description": "Use this tool to get the answer from the video language model.",
    "arguments": {
        "type": "object",
        "properties": {
            "question": {
                "type": "str",
                "description": "The question you want to use the video language model to answer."
            },
            "range": {
                "type": "str",
                "description": "The timestamp range of the video to answer the question. Use the format 'DAYX_HHMMSSFF-DAYX_HHMMSSFF'. The ending timestamp should be strictly larger than the start timestamp. The length of the range should be smaller than 10 minutes, greater than 1 second."
            }
        },
        "required": ["question", "range"]
    }
}
{
    "name": "vlm",
    "description": "Use this tool to get the answer from the vision language model.",
    "arguments": {
        "type": "object",
        "properties": {
            "question": {
                "type": "str",
                "description": "The question you want to use the vision language model to answer."
            },
            "timestamp": {
                "type": "str",
                "description": "The timestamp of the video to answer the question."
            }
        },
        "required": ["question", "timestamp"]
    }
}


For example, if you want to search for information in the RAG database, you can use the following tool:
<tool>
{
    "name": "rag",
    "arguments": {
        "level": "day",
        "keywords": ["screwdriver", "applause"],
        "start_time": "DAY1_11210217",
        "query_time": "DAY1_11220217"
    }
}
</tool>

<tool>
{
    "name": "video_llm",
    "arguments": {
        "question": "What is the answer to the question?",
        "range": "DAY1_11210217-DAY1_11220217"
    }
}
</tool>

<tool>
{
    "name": "vlm",
    "arguments": {
        "question": "What is the answer to the question?",
        "timestamp": "DAY1_11210217"
    }
}
</tool>

If the question is a multiple choice one, directly return the answer in the following format:
<answer>
{A|B|C|D}.
</answer>

"""

FORMAT_PROMPT = """
\n\nIMPORTANT: You should always think before giving an action, i.e., calling a tool or providing the answer. You can call multiple tools in multiple rounds, with each round containing either a thinking with a tool call, or a thinking with an answer. In each round, your response should ONLY be quoted by <think> & <tool>, or <think> & <answer>. DO NOT generate <information> as it should be provided by the environment.
"""

# Template dictionaries
ALPACA_TEMPLATE = {
    "instruction": "{instruction}",
    "input": "{input}",
    "output": "{output}",
    "system": "{system}"
}

TIMESTAMP_TEMPLATE = "<timestamp>{timestamp}</timestamp>"
THINKING_TEMPLATE = "<think>{thinking}</think>"
TOOL_TEMPLATE = "<tool>{tool}</tool>"
OBSERVATION_TEMPLATE = "<information>{information}</information>"
ANSWER_TEMPLATE = "<answer>{answer}</answer>"

# Global counters - consider using a class in the future for better state management
global_counter = 0
length_counter = 0


class DataProcessor:
    """Main class for processing EgoLife dataset."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def load_json_data(file_path: Union[str, Path]) -> Union[Dict, List]:
        """Load JSON data from a file with proper error handling."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading {file_path}: {e}")
    
    @staticmethod
    def save_json_data(data: Union[Dict, List], file_path: Union[str, Path]) -> None:
        """Save data to JSON file with proper formatting."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    def process_benchmark_data(self, data: Dict) -> Dict:
        """Process individual benchmark data item."""
        if not isinstance(data, dict):
            raise ValueError("Data should be a dict")
        
        # Handle query_time format
        if isinstance(data['query_time'], list):
            data['query_time'] = data['query_time'][0]
        
        try:
            timestamp = f"{data['query_time']['date']}_{data['query_time']['time']}"
        except KeyError as e:
            raise ValueError(f"Missing timestamp information: {e}")
        
        # Format question with options
        options = (f"\nA. {data['choice_a']}\nB. {data['choice_b']}\n"
                  f"C. {data['choice_c']}\nD. {data['choice_d']}")
        data['question'] = (f"{data['question']} "
                           f"{TIMESTAMP_TEMPLATE.format(timestamp=timestamp)} \n{options}")
        
        return data
    
    def extract_observation(self, tool_name: str, obs_str: str) -> str:
        """Extract observation content based on tool type."""
        try:
            obs = ast.literal_eval(obs_str)
            
            if tool_name == "rag":
                return obs["response"].get("relevant_content", 
                                        obs["response"]["generated_text"])
            elif tool_name in ["video_llm", "vlm"]:
                return obs["answer"]
            else:
                raise ValueError(f"Unknown tool name: {tool_name}")
        except (ValueError, KeyError, SyntaxError):
            return obs_str
    
    def format_tool_data(self, tool: Dict) -> Dict:
        """Format tool data structure."""
        return {
            "name": tool["name"],
            "arguments": tool["arguments"]
        }
    
    def format_template(self, template: Dict, **kwargs) -> Dict:
        """Format a template with provided kwargs."""
        return {k: v.format(**kwargs) for k, v in template.items()}
    
    def validate_data_quality(self, data: Dict) -> bool:
        """Validate data quality before processing."""
        # Check COT length
        if len(data["cot"]) > MAX_COT_LENGTH:
            print(f"Skipping data with cot length > {MAX_COT_LENGTH}: {data['ID']}")
            return False
        
        # Check answer correctness
        if data['cot'][-1]['answer'] != data['ground_truth']:
            print(f"Skipping data with wrong answer: {data['ID']}")
            return False
        
        return True

    def build_conversation_history(self, cot_data: List[Dict], current_step: int) -> List[List[str]]:
        """Build conversation history up to current step."""
        history = []
        
        for step in range(current_step):
            if step == 0:
                # First step uses question as input
                traj = [cot_data[step]["question"], cot_data[step]["response"]]
            else:
                # Subsequent steps use observation as input
                traj = [cot_data[step-1]["observation"], cot_data[step]["response"]]
            history.append(traj)
        
        return history

    def convert_to_alpaca_format(self, json_files: List[str], identity: str, 
                               split: str, stage: str) -> None:
        """Convert data to Alpaca format for training."""
        global global_counter, length_counter
        
        data_list = []
        json_data_list = []
        identity_name = IDENTITY_MAPPING.get(identity, identity)
        
        print(f"Processing data for identity: {identity} ({identity_name})")
        print(f"Files to process: {len(json_files)}")
        print(f"Current global counter: {global_counter}")
        
        for file_path in json_files:
            try:
                data = self.load_json_data(file_path)
                
                if not self.validate_data_quality(data):
                    length_counter += 1
                    continue
                
                data["ID"] = global_counter
                json_data_list.append(data)
                
                # Process each step in the chain of thought
                processed_steps = self._process_cot_steps(data)
                data_list.extend(processed_steps)
                global_counter += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        self._save_processed_data(data_list, json_data_list, identity, split, stage)
        self._print_processing_summary(data_list, json_data_list)
    
    def _process_cot_steps(self, data: Dict) -> List[Dict]:
        """Process individual chain-of-thought steps."""
        steps_data = []
        operations = []
        observations = []
        
        # Generate operations and observations for each step
        for i, step in enumerate(data["cot"]):
            try:
                if i == len(data["cot"]) - 1:  # Last step
                    op = (f"{THINKING_TEMPLATE.format(thinking=step['thought'])} "
                         f"{ANSWER_TEMPLATE.format(answer=step['answer'])}")
                else:
                    op = (f"{THINKING_TEMPLATE.format(thinking=step['thought'])} "
                         f"{TOOL_TEMPLATE.format(tool=self.format_tool_data(step['tool']))}")
                    observations.append(
                        OBSERVATION_TEMPLATE.format(information=str(step["observation"]))
                    )
                
                operations.append(op)
                
                # Create training data for this step
                step_data = self._create_step_data(data, operations, observations, i)
                steps_data.append(step_data)
                
            except Exception as e:
                print(f"Error processing step {i} in {data.get('ID', 'unknown')}: {e}")
                continue
        
        return steps_data
    
    def _create_step_data(self, data: Dict, operations: List[str], 
                         observations: List[str], step_index: int) -> Dict:
        """Create training data for a specific step."""
        # Build history for this step
        history = []
        for j in range(step_index):
            if j == 0:
                traj = [data["question"], operations[j]]
            else:
                traj = [observations[j-1], operations[j]]
            history.append(traj)
        
        # Create Alpaca format data
        if step_index == 0:
            step_data = self.format_template(ALPACA_TEMPLATE, **{
                "instruction": data["question"],
                "input": "",
                "output": operations[step_index],
                "system": PROMPT_TEMPLATE + FORMAT_PROMPT
            })
        else:
            step_data = self.format_template(ALPACA_TEMPLATE, **{
                "instruction": observations[step_index-1],
                "input": "",
                "output": operations[step_index],
                "system": PROMPT_TEMPLATE + FORMAT_PROMPT
            })
        
        step_data["history"] = history
        return step_data
    
    def _save_processed_data(self, data_list: List[Dict], json_data_list: List[Dict],
                           identity: str, split: str, stage: str) -> None:
        """Save processed data to files."""
        sft_file = self.output_dir / f"{identity}_{split}_{stage}.json"
        raw_file = self.output_dir / f"{identity}_{split}_raw.json"
        
        self.save_json_data(data_list, sft_file)
        self.save_json_data(json_data_list, raw_file)
    
    def _print_processing_summary(self, data_list: List[Dict], json_data_list: List[Dict]) -> None:
        """Print processing summary statistics."""
        print(f"Size of the sft dataset: {len(data_list)}")
        print(f"Size of the rl dataset: {len(json_data_list)}")
        print(f"Skipped {length_counter} data with cot length > {MAX_COT_LENGTH}")


def concatenate_json_files(files: List[str], output_file: str) -> List[Dict]:
    """Concatenate multiple JSON files into a single file."""
    all_data = []
    
    for file_path in files:
        print(f"Processing: {file_path}")
        try:
            data = DataProcessor.load_json_data(file_path)
            
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    DataProcessor.save_json_data(all_data, output_file)
    print(f"Concatenated {len(all_data)} items into {output_file}")
    return all_data

def concatenate_json_data(all_data: List[Dict], output_file: str) -> List[Dict]:
    """Process and concatenate benchmark data."""
    processor = DataProcessor(os.path.dirname(output_file))
    
    for item in all_data:
        processor.process_benchmark_data(item)
    
    DataProcessor.save_json_data(all_data, output_file)
    print(f"Concatenated {len(all_data)} items into {output_file}")
    return all_data

def concatenate_parquet_files(files: List[str], output_file: str) -> None:
    """Concatenate multiple parquet files into a single file."""
    datasets = []
    for file_path in files:
        try:
            data = load_dataset("parquet", data_files=file_path, split="train")
            datasets.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if datasets:
        combined_data = concatenate_datasets(datasets)
        combined_data.to_parquet(output_file)
        print(f"Concatenated {len(combined_data)} items into {output_file}")

def convert_to_etos_format(jsonl_path: str, split: str, stage: str) -> str:
    """Convert JSONL to Parquet format with specific schema."""
    try:
        ds = load_dataset("json", data_files=jsonl_path)
        
        def map_function(example, index):
            answer = example['ground_truth'] if split == 'train' else example['answer']
            original_question = example['question']
            
            return {
                **example,
                'question': PROMPT_TEMPLATE + FORMAT_PROMPT + original_question,
                'data_source': 'multi-choice',
                'prompt': [{'role': 'user', 'content': example['question']}],
                'golden_answer': [answer],
                'ability': 'fact-reasoning',
                'reward_model': {'ground_truth': {'target': answer}, 'style': 'rule'},
                'extra_info': {'index': index, 'split': split}
            }
        
        ds = ds.map(map_function, with_indices=True)['train']
        
        # Determine output path
        if jsonl_path.endswith('.jsonl'):
            parquet_path = jsonl_path.replace('_raw.jsonl', f'_{stage}.parquet')
        elif jsonl_path.endswith('.json'):
            parquet_path = jsonl_path.replace('_raw.json', f'_{stage}.parquet')
        else:
            raise ValueError(f"Invalid file extension: {jsonl_path}")
        
        ds.to_parquet(parquet_path)
        return parquet_path
        
    except Exception as e:
        print(f"Error converting {jsonl_path} to ETOS format: {e}")
        raise

def fix_json_types(input_path: str) -> None:
    """Fix data types in JSON file."""
    try:
        data = DataProcessor.load_json_data(input_path)
        print(f"Loaded {input_path}")
        
        for item in data:
            # Fix choice types
            choice_keys = ['choice_a', 'choice_b', 'choice_c', 'choice_d', 
                          'choice_a_chinese', 'choice_b_chinese', 'choice_c_chinese', 'choice_d_chinese']
            
            for key in choice_keys:
                if key in item and not isinstance(item[key], str):
                    item[key] = str(item[key])
            
            # Fix query_time format
            if isinstance(item["query_time"], list):
                item["query_time"] = item["query_time"][0]
                print(f"Fixed query_time for entry: {item['ID']}")
            
            # Fix last_time boolean
            if item["last_time"] == "N/A":
                item["last_time"] = False
                print(f"Fixed bool last_time for entry: {item['ID']}")
        
        DataProcessor.save_json_data(data, input_path)
        print(f"Fixed {input_path}")
        
    except Exception as e:
        print(f"Error fixing JSON types in {input_path}: {e}")
        raise

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(description="Process EgoLife data for training")
    parser.add_argument("--home_dir", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="train_data")
    parser.add_argument("--benchmark_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_source", type=str, default="EgoLife")
    parser.add_argument("--format", type=str, choices=["alpaca", "glaive"], default="alpaca")
    parser.add_argument("--random_split", action="store_true")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--stage", type=str, choices=["sft", "rl", "all"], default="sft")
    return parser

def main():
    """Main function to process data."""
    args = create_argument_parser().parse_args()
    
    # Set default output directory if not provided
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d')
        args.output_dir = os.path.join("data", f"postprocess_{timestamp}")
    
    processor = DataProcessor(args.output_dir)
    
    # Process benchmark data for test split
    benchmark_data = []
    for identity in IDENTITIES:
        benchmark_file = os.path.join(
            args.benchmark_dir, 
            f"{identity}_{IDENTITY_MAPPING[identity]}.json"
        )
        
        try:
            bench_data = processor.load_json_data(benchmark_file)
            benchmark_data.extend(bench_data)
            
            if len(bench_data) != EXPECTED_BENCHMARK_ITEMS:
                print(f"Warning: Expected {EXPECTED_BENCHMARK_ITEMS} benchmark items, "
                      f"got {len(bench_data)} for {identity}")
        except Exception as e:
            print(f"Error loading benchmark data for {identity}: {e}")
            continue
        
        # Process training data if split is train
        if args.split == "train":
            bench_ids = {item["ID"] for item in bench_data}
            
            dp_files = sorted(
                glob.glob(os.path.join(args.data_dir, identity, "*.json")), 
                key=lambda x: int(Path(x).stem)
            )
            
            # Filter out benchmark files from training data
            dp_train_files = [
                file for file in dp_files 
                if int(Path(file).stem) not in bench_ids
            ]
            
            # Process based on stage
            if args.stage in ["sft", "all"]:
                processor.convert_to_alpaca_format(
                    dp_train_files, identity, args.split, "sft"
                )
            
            if args.stage in ["rl", "all"]:
                output_path = os.path.join(args.output_dir, f"{identity}_{args.split}_raw.json")
                if args.stage == "rl":
                    convert_to_etos_format(output_path, args.split, "rl")
                else:  # "all"
                    concatenate_json_files(dp_train_files, output_path)
                    convert_to_etos_format(output_path, args.split, "sft")
    
    # Concatenate results based on split and stage
    if args.split == "train":
        _handle_train_concatenation(args)
    elif args.split == "test":
        _handle_test_processing(args, benchmark_data)

def _handle_train_concatenation(args):
    """Handle concatenation for training data."""
    if args.stage in ["sft", "all"]:
        sft_files = glob.glob(os.path.join(args.output_dir, "*_train_sft.json"))
        if sft_files:
            concatenate_json_files(
                sft_files, 
                os.path.join(args.output_dir, "train_sft.json")
            )
            print(f"Successfully concatenated {len(sft_files)} sft files for train set")
    
    if args.stage in ["rl", "all"]:
        rl_files = glob.glob(os.path.join(args.output_dir, "*_train_rl.parquet"))
        if rl_files:
            concatenate_parquet_files(
                rl_files, 
                os.path.join(args.output_dir, "train_rl.parquet")
            )
            print(f"Successfully concatenated {len(rl_files)} rl files for train set")

def _handle_test_processing(args, benchmark_data):
    """Handle processing for test data."""
    output_path = os.path.join(args.output_dir, "test.json")
    
    # Load and concatenate all benchmark files
    if not benchmark_data:
        benchmark_files = [f for f in os.listdir(args.output_dir) if f.endswith('.json')]
        for file in benchmark_files:
            try:
                data = DataProcessor.load_json_data(os.path.join(args.output_dir, file))
                benchmark_data.extend(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    if benchmark_data:
        concatenate_json_data(benchmark_data, output_path)
        fix_json_types(output_path)
        convert_to_etos_format(output_path, "test", "rl")
        print("Successfully processed test data")


if __name__ == "__main__":
    main()

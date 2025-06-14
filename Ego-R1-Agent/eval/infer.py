import transformers
import torch
import random
from datasets import load_dataset
import requests
import re
import json
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional
from tqdm import tqdm
from openai import OpenAI

import utils.process as process
import utils.constants as constants
from vllm import LLM, SamplingParams

# Configuration for local OpenAI-compatible API (if using)
openai_api_key = 'EMPTY'
openai_api_base = 'http://localhost:23332/v1'

@dataclass
class ScriptArgs:
    """Command line arguments for the inference script.
    
    This class defines all configurable parameters for model inference including:
    - Model selection and path
    - Dataset configuration
    - Generation parameters (temperature, top_p, etc.)
    - LoRA fine-tuning options
    """
    model_name_or_path: Optional[str] = field(
        default='Ego-R1/Ego-R1-Agent-3B',
        metadata={"help": "Model name or path"}
    )
    dataset: Optional[str] = field(
        default='./data/test.parquet',
        metadata={"help": "Dataset name"}
    )
    data_end: Optional[int] = field(
        default=-1,
        metadata={"help": "Data end"}
    )

    # Generation parameters
    temperature: Optional[float] = field(
        default=0.6,
        metadata={"help": "Temperature for the model"}
    )
    top_p: Optional[float] = field(
        default=0.95,
        metadata={"help": "Top-p for the model"}
    )
    max_turns: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of turns"}
    )
    max_new_tokens: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum number of new tokens"}
    )

    # LoRA fine-tuning options
    use_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Use lora"}
    )
    base_model: Optional[str] = field(
        default='Qwen/Qwen2.5-3B-Instruct',
        metadata={"help": "Base model"}
    )


def main():
    """Main inference function that:
    1. Loads the model and dataset
    2. Processes each question through multiple reasoning turns
    3. Evaluates model responses against ground truth
    4. Tracks and reports accuracy
    """
    # Parse command line arguments
    parser = HfArgumentParser(ScriptArgs)
    args = parser.parse_args_into_dataclasses()[0]

    # Initialize model and system prompt
    system_prompt = constants.prompt + constants.format_prompt
    client = LLM(model=args.model_name_or_path, tokenizer=args.model_name_or_path, gpu_memory_utilization=0.5, tensor_parallel_size=1)

    # Load and prepare dataset
    ds = load_dataset('parquet', data_files={'test': args.dataset}, split='test')
    if args.data_end and args.data_end > 0:
        ds = ds.select(range(args.data_end))
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Special tokens for Qwen2.5 models
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    
    # Initialize accuracy tracking
    acc = 0
    for i, item in enumerate(tqdm(ds, desc='Evaluating...')):
        # Extract question and ground truth
        question = item['question'][len(system_prompt):]
        gt = item['reward_model']['ground_truth']['target']

        turn_cnt = 0
        chat_history = [{'role': 'system', 'content': system_prompt}]

        print('\n\n################# [Start Reasoning + Tool Calling] ##################\n\n')
        print('='*20, 'system', '='*20)
        print(system_prompt)
        user_input = f"{question}"
        
        # Multi-turn reasoning loop
        while True:
            if turn_cnt >= args.max_turns:
                print(f'Turns exceed the maximum number of turns: {args.max_turns}.')
                break

            # Add user input to chat history and generate response
            chat_history.append({"role": "user", "content": user_input})
            print('='*20, 'user_input', '='*20)
            print(user_input)
            client_input = tokenizer.apply_chat_template(chat_history, add_generation_prompt=True, tokenize=False)
            if turn_cnt == args.max_turns-1:
                client_input += constants.stop_prompt

            # Generate model response
            output_message = client.generate(
                [client_input],
                sampling_params=SamplingParams(
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    n=1
                )
            )[0].outputs[0].text
            if turn_cnt == args.max_turns-1:
                output_message = constants.stop_prompt + output_message

            # Update chat history with model response
            chat_history.append({"role": "assistant", "content": output_message})
                
            output_text = output_message
            print('='*20, 'output_text', '='*20)
            print(output_text)
            
            # Check if model has provided final answer
            if '<answer>' in output_message and '</answer>' in output_message:
                break

            # Process tool calls if present
            tmp_query = process.get_query(output_text)
            tmp_query = process.parse_tool_call(tmp_query)
            if tmp_query:
                print(f'Tool call: "{tmp_query}"...')
                search_results = process.tool_call(tmp_query, item['identity'])
            else:
                search_results = ''

            # Prepare next turn with search results
            search_text = f'<information>{search_results}</information>'
            user_input = search_text
            turn_cnt += 1

        # Evaluate response against ground truth
        score = process.compute_score(output_text, gt)
        
        # Log results
        if score == 1:
            with open(f'correct_questions_{args.model_name_or_path.split("/")[-1]}_maxturn{args.max_turns}_{args.data_end}.txt', 'a') as f:
                f.write(f'{question}\n')
        else:
            with open(f'incorrect_questions_{args.model_name_or_path.split("/")[-1]}_maxturn{args.max_turns}_{args.data_end}.txt', 'a') as f:
                f.write(f'{question}\n\n{chat_history}\n\n')
        acc += score
        print(f"Current accuracy: {acc / (i + 1)}")

    # Print final accuracy
    print(f'Accuracy: {acc / len(ds)}')

if __name__ == "__main__":
    main()


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
from openai import OpenAI, AzureOpenAI  
from datetime import datetime
import os

import utils.process as process
import utils.constants as constants
from vllm import LLM, SamplingParams

HAS_INST = False
endpoint = os.getenv("ENDPOINT_URL", "https://kcaudio.openai.azure.com/")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")  

summ_client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2025-01-01-preview",
)

openai_api_key = 'EMPTY'
openai_api_base = 'http://localhost:23332/v1'

datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("results/egoschema_debug")
os.makedirs(save_dir, exist_ok=True)

@dataclass
class ScriptArgs:
    model_name_or_path: Optional[str] = field(
        default='Ego-R1/qwen-25-3b-it-sft4500-len8192-rl-bs32-gs20',
        metadata={"help": "Model name or path"}
    )
    dataset: Optional[str] = field(
        default='./benchmarks/video-mme-long.parquet',
        metadata={"help": "Dataset name"}
    )
    data_start: Optional[int] = field(
        default=0,
        metadata={"help": "Data start"}
    )
    data_end: Optional[int] = field(
        default=-1,
        metadata={"help": "Data end"}
    )
    bench_name: Optional[str] = field(
        default='video-mme-long',
        metadata={"help": "Benchmark name"}
    )
    # generation
    temperature: Optional[float] = field(
        default=0.6,
        metadata={"help": "Temperature for the model"}
    )
    top_p: Optional[float] = field(
        default=0.95,
        metadata={"help": "Top-p for the model"}
    )
    max_turns: Optional[int] = field(
        default=5,
        metadata={"help": "Maximum number of turns"}
    )
    max_new_tokens: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum number of new tokens"}
    )

    # lora
    use_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Use lora"}
    )
    base_model: Optional[str] = field(
        default='Qwen/Qwen2.5-3B-Instruct',
        metadata={"help": "Base model"}
    )

def main():
    parser = HfArgumentParser(ScriptArgs)
    args = parser.parse_args_into_dataclasses()[0]
    
    print(f'Currently evaluting {args.bench_name}...Loading dataset from {args.dataset}...')
    system_prompt = constants.prompt + constants.format_prompt
    client = LLM(model=args.model_name_or_path, tokenizer=args.model_name_or_path, gpu_memory_utilization=0.5, tensor_parallel_size=1)

    ds = load_dataset('parquet', data_files={'test': args.dataset}, split='test')
    if args.data_end and args.data_end > 0:
        ds = ds.select(range(args.data_end))
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    acc = 0
    cnt = 0
    results = []
    for i, item in enumerate(tqdm(ds, desc='Evaluating...')):
        if i < args.data_start:
            continue
        cnt += 1
        if HAS_INST:
            question = item['question'][len(system_prompt):]
        else:
            question = item['question']
            
        gt = item['golden_answer']

        turn_cnt = 0

        if args.bench_name == 'egoschema':
            tmp_system = constants.prompt_egoschema + constants.format_prompt
            chat_history = [{'role': 'system', 'content': tmp_system}]
        elif args.bench_name == 'video-mme-long':
            tmp_system = constants.prompt_mme + constants.format_prompt
            chat_history = [{'role': 'system', 'content': tmp_system}]
        else:
            tmp_system = system_prompt
            chat_history = [{'role': 'system', 'content': tmp_system}]
            
        print('='*20, 'system', '='*20)
        print(tmp_system)

        print('\n\n################# [Start Reasoning + Tool Calling] ##################\n\n')
        user_input = f"You should answer the question in {args.max_turns} turns. {question}"
        while True:
            if turn_cnt >= args.max_turns:
                print(f'Turns exceed the maximum number of turns: {args.max_turns}.')
                break
                
            chat_history.append({"role": "user", "content": user_input})
            print('='*20, 'user_input', '='*20)
            print(user_input)
            client_input = tokenizer.apply_chat_template(chat_history, add_generation_prompt=True, tokenize=False)
            
            
            if turn_cnt == args.max_turns-1:
                if args.bench_name == 'video-mme-long':
                    client_prompt = """You are given some information and a chain-of-thought reasoning process, with actions made and observations from the environment. Given these information, try to answer the MCQ question in A|B|C|D format. You should only answer one option. For example, if the answer to this question is A, you should only return ```<answer>A</answer>```."""
                elif args.bench_name == 'egoschema':
                    client_prompt = """You are given some information and a chain-of-thought reasoning process, with actions made and observations from the environment. Given these information, try to answer the MCQ question in A|B|C|D|E format. You should only answer one option. For example, if the answer to this question is E, you should only return ```<answer>E</answer>```."""
                ch_prompt = f"The chat history is as follows:\n{chat_history}\n"
                messages = [
                    {"role": "system", "content": client_prompt},
                    {"role": "user", "content": ch_prompt},
                    {"role": "user", "content": f"Question: {question}"}
                ]
                for _ in range(3):
                    try:
                        output_message = summ_client.chat.completions.create(  
                            model=deployment,
                            messages=messages,
                            max_tokens=800,  
                            temperature=1,  
                            top_p=1,  
                            frequency_penalty=0,  
                            presence_penalty=0,
                            stop=None,  
                            stream=False
                        ).choices[0].message.content
                        assert '<answer>' in output_message and '</answer>' in output_message
                        break
                    except Exception as e:
                        print(f'Error: {e}')
                        if _ == 2:  # Last iteration (0, 1, 2)
                            if "<answer>" in output_message:
                                break
                            else:
                                print("Max retries reached, using fallback answer")
                                output_message = "<answer>A</answer>"  # or some default
                                break
            else:
                output_message = client.generate(
                    [client_input],
                    sampling_params=SamplingParams(
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        n=1
                    )
                )[0].outputs[0].text

            chat_history.append({"role": "assistant", "content": output_message})
            
            # old answer
            output_text = output_message
            print('='*20, 'output_text', '='*20)
            if turn_cnt == args.max_turns-1:
                print('='*20, 'summarized_output_text', '='*20)
            print(output_text)
            if '<answer>' in output_message and '</answer>' in output_message:
                if turn_cnt != args.max_turns-1:
                    if args.bench_name == 'video-mme-long': 
                        client_prompt = """You are given some information and a chain-of-thought reasoning process, with actions made and observations from the environment. Given these information, try to answer the MCQ question in A|B|C|D format. You should only answer one option. For example, if the answer to this question is A, you should only return ```<answer>A</answer>```."""
                    elif args.bench_name == 'egoschema':
                        client_prompt = """You are given some information and a chain-of-thought reasoning process, with actions made and observations from the environment. Given these information, try to answer the MCQ question in A|B|C|D|E format. You should only answer one option. For example, if the answer to this question is E, you should only return ```<answer>E</answer>```."""
                    ch_prompt = f"The chat history is as follows:\n{chat_history[-1]}\n"
                    messages = [
                        {"role": "system", "content": client_prompt},
                        {"role": "user", "content": ch_prompt},
                        {"role": "user", "content": f"Question: {question}"}
                    ]
                    for _ in range(3):
                        try:
                            output_message = summ_client.chat.completions.create(  
                                model=deployment,
                                messages=messages,
                                max_tokens=800,  
                                temperature=1,  
                                top_p=1,  
                                frequency_penalty=0,  
                                presence_penalty=0,
                                stop=None,  
                                stream=False
                            ).choices[0].message.content
                            assert '<answer>' in output_message and '</answer>' in output_message
                            break
                            
                        except Exception as e:
                            print(f'Error: {e}')
                            if _ == 2:  # Last iteration (0, 1, 2)
                                if "<answer>" in output_message:
                                    break
                                else:
                                    print("Max retries reached, using fallback answer")
                                    output_message = "<answer>A</answer>"  # or some default
                                    break

                    print('='*20, 'summarized_output_message', '='*20)
                    print(output_message)

                break
            
            ## 20250604 add pre_tool_prompt ##
            if args.bench_name == 'egoschema' and turn_cnt == 0:
                estimated_range = "DAY1_00000000-DAY1_00095900"
                output_text = "<think>I should think about the question first, and call video_llm to get the information.</think>\n" + constants.pre_tool_prompt.format(question=question, range=estimated_range)
            ## 20250604 add pre_tool_prompt ##
            
            tmp_query = process.get_query(output_text)
            tmp_query = process.parse_tool_call(tmp_query)
            if tmp_query:
                # video-mme-long append videoID to the query
                if args.bench_name == 'video-mme-long':
                    tmp_query['vid_id'] = item['videoID']
                elif args.bench_name == 'egoschema':
                    tmp_query['vid_id'] = item['video_idx']
                print(f'Tool call: "{tmp_query}"...')
                search_results = process.tool_call(tmp_query, args.bench_name)
            else:
                search_results = ''

            search_text = f'<information>{search_results}</information>'
            user_input = search_text
            # chat_history.append({"role": "user", "content": search_text})
            turn_cnt += 1

        if args.bench_name == 'egoschema':
            results.append({
                "video_idx": item['video_idx'],
                "answer": output_text
            })
        score = process.compute_score(output_text, gt)
        print("ground truth: ", gt, "current score: ", score)
        
    
        # # save the correct question
        # correct_file = f'results/correct/correct_questions_{args.bench_name}_{args.model_name_or_path.split("/")[-1]}_maxturn{args.max_turns}_{args.data_end}.txt'
        # incorrect_file = f'results/incorrect/incorrect_questions_{args.bench_name}_{args.model_name_or_path.split("/")[-1]}_maxturn{args.max_turns}_{args.data_end}.txt'
        # os.makedirs('results/correct', exist_ok=True)
        # os.makedirs('results/incorrect', exist_ok=True)

        acc += score
        print(f"Current accuracy: {acc / cnt}")
        
    if args.bench_name == 'egoschema':
        # save the results after the last item
        with open(f'{save_dir}/egoschema_{args.model_name_or_path.split("/")[-1]}_maxturn{args.max_turns}_s{args.data_start}_e{args.data_end}_debug_{datetime}.json', 'w') as f:
            json.dump(results, f, indent=4)
    print(f'Accuracy: {acc / cnt}')

if __name__ == "__main__":
    main()


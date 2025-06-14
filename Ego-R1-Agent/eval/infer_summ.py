import transformers
import torch
import random
from datasets import load_dataset
import requests
import re
import json
import os
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from typing import Optional
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI  

import utils.process as process
import utils.constants as constants
from vllm import LLM, SamplingParams

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

@dataclass
class ScriptArgs:
    model_name_or_path: Optional[str] = field(
        default='.25-3b-it-sft4500-len8192-rl-bs4-20250520/actor/global_step_145',
        metadata={"help": "Model name or path"}
    )
    dataset: Optional[str] = field(
        default='.ot10_yn/test.parquet',
        metadata={"help": "Dataset name"}
    )
    data_end: Optional[int] = field(
        default=-1,
        metadata={"help": "Data end"}
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
        default=12,
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
    result_dir: Optional[str] = field(
        default='results',
        metadata={"help": "Result directory"}
    )

def main():
    parser = HfArgumentParser(ScriptArgs)
    args = parser.parse_args_into_dataclasses()[0]

    system_prompt = constants.prompt + constants.format_prompt
    client = LLM(model=args.model_name_or_path, tokenizer=args.model_name_or_path, gpu_memory_utilization=0.5, tensor_parallel_size=1)

    ds = load_dataset('parquet', data_files={'test': args.dataset}, split='test')
    if args.data_end and args.data_end > 0:
        ds = ds.select(range(args.data_end))
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    acc = 0
    for i, item in enumerate(tqdm(ds, desc='Evaluating...')):
        # if only A1 is used, skip the rest
        # if not "A1" in item["identity"]:
        #     continue
        question = item['question'][len(system_prompt):]
        gt = item['reward_model']['ground_truth']['target']

        turn_cnt = 0

        chat_history = [{'role': 'system', 'content': system_prompt}]


        print('\n\n################# [Start Reasoning + Tool Calling] ##################\n\n')
        print('='*20, 'system', '='*20)
        print(system_prompt)
        user_input = f"{question}"
        while True:
            if turn_cnt >= args.max_turns:
                print(f'Turns exceed the maximum number of turns: {args.max_turns}.')
                break

            chat_history.append({"role": "user", "content": user_input})
            print('='*20, 'user_input', '='*20)
            print(user_input)
            client_input = tokenizer.apply_chat_template(chat_history, add_generation_prompt=True, tokenize=False)
            
            if turn_cnt == args.max_turns-1:
                client_prompt = """You are given some information and a chain-of-thought reasoning process, with actions made and observations from the environment. Given these information, try to answer the MCQ question in A|B|C|D format. You should only answer one option. For example, if the answer to this question is A, you should only return ```<answer>A</answer>```."""
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
                        print(f'Error: {e} {output_message}')
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
            print(output_text)
            if '<answer>' in output_message and '</answer>' in output_message:
                
                # for those haven't been summarized by the model, but the model get the answer itself
                if turn_cnt != args.max_turns-1:
                    client_prompt = """You are given the full reasoning path with information about the actions made and observations from the environment. Given these information, try to answer the MCQ question in A|B|C|D format. You should only answer one option. For example, if the answer to this question is A, you should only return ```<answer>A</answer>```."""
                    ch_prompt = f"The chat history is as follows:\n{chat_history[:-1]}\n"
                    messages = [
                        {"role": "system", "content": client_prompt},
                        {"role": "user", "content": ch_prompt},
                        {"role": "user", "content": f"## Question: {question}"}
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
                            
                            # new answer, update the output_text
                            output_text = output_message
                            chat_history.append({"role": "assistant", "content": f"## Revised Summarized Answer: {output_text}"})
                            break
                        
                        except Exception as e:
                            print(f'Error: {e} {output_message}')
                            if _ == 2:
                                print("Max retries reached, using fallback answer")
                                output_text = "<answer>A</answer>"
                                break
                
                break
            
            # get the tool call
            tmp_query = process.get_query(output_text)
            tmp_query = process.parse_tool_call(tmp_query)
            if tmp_query:
                print(f'Identity: {item["identity"]} \nTool call: "{tmp_query}"...')
                search_results = process.tool_call(tmp_query, item['identity'])
            else:
                search_results = ''

            search_text = f'<information>{search_results}</information>'
            user_input = search_text
            # chat_history.append({"role": "user", "content": search_text})
            turn_cnt += 1

        score = process.compute_score(output_text, gt)
        
        # save the correct question
        ## make the save directory
        save_dir = f'{args.result_dir}/{args.model_name_or_path.split("/")[-1]}_maxturn{args.max_turns}_{args.data_end}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        correct_dir = f'{save_dir}/correct_questions'
        incorrect_dir = f'{save_dir}/incorrect_questions'
        if not os.path.exists(correct_dir):
            os.makedirs(correct_dir)
        if not os.path.exists(incorrect_dir):
            os.makedirs(incorrect_dir)
        
        ## save the correct question
        if score == 1:
            with open(f'{correct_dir}/{i}.txt', 'w') as f:
                f.write(f'Identity: {item["identity"]}\n')
                f.write(f'Question: {question}\n')
                f.write(f'Ground Truth: {gt}\n')
                f.write(f'Score: {score}\n')
                f.write(f'Chat History: {chat_history}\n')
        else:
            ## save the incorrect question
            with open(f'{incorrect_dir}/{i}.txt', 'w') as f:
                f.write(f'Identity: {item["identity"]}\n')
                f.write(f'Question: {question}\n')
                f.write(f'Ground Truth: {gt}\n')
                f.write(f'Score: {score}\n')
                f.write(f'Chat History: {chat_history}\n')
        acc += score
        print(f"Current accuracy: {acc / (i + 1)}")

    print(f'Accuracy: {acc / len(ds)}')

if __name__ == "__main__":
    main()


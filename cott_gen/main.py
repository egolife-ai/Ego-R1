import asyncio
import json
import os
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import ModelFamily
from prompts import thought_prompt, sys_prompt, identity_prompt
from tools import *
from utils import extract_json, identity_mapping_dict, setup_logging_and_config, process_qa, rag_url_dict
from datetime import datetime

# max steps for the agent to generate the answer
MAX_STEPS = 20


class AgentResponse:
    """Class to represent agent response structure."""
    def __init__(self, thoughts: str, response: str):
        self.thoughts = thoughts
        self.response = response


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run EgoLife QA agent with chain-of-thought reasoning"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4.1",
        help="Model to use for the agent"
    )
    parser.add_argument(
        "--api_version", 
        type=str, 
        default="2024-09-01-preview",
        help="API version for the model"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="./egor1-bench/QA-egolife/",
        help="Path to the data directory"
    )
    parser.add_argument(
        "--identity", 
        type=str, 
        default="A1",
        help="Identity to use for the agent"
    )
    
    # Output configuration
    parser.add_argument(
        "--result_dir", 
        type=str, 
        default="./results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="./logs",
        help="Directory to save logs"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="./cache",
        help="Directory for caching"
    )
    
    # Processing options
    parser.add_argument(
        "--explicit_answer", 
        action="store_true",
        help="Use explicit answer termination"
    )
    parser.add_argument(
        "--observation_type", 
        type=str, 
        default="all_actions", 
        choices=["single", "all", "all_actions", "null"],
        help="Type of observation to include in prompts"
    )
    
    # Resume and specific data options
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume processing from error files"
    )
    parser.add_argument(
        "--gen_specific_data", 
        action="store_true",
        help="Process only specific data IDs"
    )
    parser.add_argument(
        "--specific_data_path", 
        type=str, 
        default="./data_statistics/error_list_results_aobs_gpt-41_A1.txt",
        help="Path to specific data list file (.txt or .json)"
    )
    
    return parser.parse_args()


def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from file with error handling."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to load {file_path}: {e}")


def load_specific_data_ids(file_path: str) -> List[int]:
    """Load specific data IDs from a text or JSON file."""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            data_str = f.readlines()[0].strip()
            # Handle both comma-separated and bracket-enclosed formats
            data_str = data_str.strip("[]")
            return [int(x.strip()) for x in data_str.split(",") if x.strip()]
    
    elif file_path.endswith(".json"):
        error_data = load_json_data(file_path)
        return [int(item["ID"]) for item in error_data]
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def load_resume_data_ids(identity: str) -> List[int]:
    """Load data IDs that need to be resumed from error files."""
    error_ids = []
    
    # Load NA errors
    na_error_path = f"data_gen/errors/error_list_na_{identity}.json"
    if os.path.exists(na_error_path):
        try:
            na_errors = load_json_data(na_error_path)
            error_ids.extend([int(d["ID"]) for d in na_errors])
        except Exception as e:
            print(f"Warning: Could not load NA errors from {na_error_path}: {e}")
    
    # Load no-answer errors (incomplete processing)
    no_answer_path = f"data_gen/errors/error_list_no_answer_{identity}.json"
    if os.path.exists(no_answer_path):
        try:
            no_answer_errors = load_json_data(no_answer_path)
            error_ids.extend([
                int(error_d["ID"]) 
                for error_d in no_answer_errors 
                if len(error_d.get("cot", [])) < MAX_STEPS
            ])
        except Exception as e:
            print(f"Warning: Could not load no-answer errors from {no_answer_path}: {e}")
    
    return list(set(error_ids))  # Remove duplicates


def filter_data_by_ids(data: List[Dict], target_ids: List[int]) -> List[Dict]:
    """Filter data to only include items with IDs in target_ids."""
    return [item for item in data if item["ID"] in target_ids]


def setup_model_client(args: argparse.Namespace) -> AzureOpenAIChatCompletionClient:
    """Set up the model client based on arguments."""
    endpoint, deployment, subscription_key = setup_logging_and_config(args.model)
    
    if args.model == "gpt-4.1":
        return AzureOpenAIChatCompletionClient(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            model="gpt-41",
            api_version="2025-01-01-preview",
            api_key=subscription_key,
            model_info={
                "family": ModelFamily.GPT_41,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "vision": False,
            }
        )
    else:
        return AzureOpenAIChatCompletionClient(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            model=args.model,
            api_version=args.api_version,
            api_key=subscription_key,
        )


def get_tools(args: argparse.Namespace) -> List:
    """Get the appropriate tools based on arguments."""
    if args.explicit_answer:
        return [rag, video_llm, vlm, terminate_explicit]
    else:
        return [rag, video_llm, vlm, terminate]


def get_system_prompt(args: argparse.Namespace) -> str:
    """Get the appropriate system prompt based on version."""
    return sys_prompt + f"\n{identity_prompt.format(identity=f'{args.identity}_{identity_mapping_dict[args.identity]}')}"


def create_resume_prompt(dp: Dict, base_prompt: List[TextMessage]) -> List[TextMessage]:
    """Create a prompt for resuming from previous CoT steps."""
    prompt = base_prompt.copy()
    
    if dp.get("cot"):
        prompt.append(TextMessage(
            source="assistant", 
            content=f'Previous observations: {dp["cot"]}'
        ))
        prompt.append(TextMessage(
            source="user",
            content=(
                "Now you are given the previous actions and observations you have made before, "
                "continue to try your best to answer the question using different tools. "
                f"You must provide an answer to the question before step {MAX_STEPS}."
            )
        ))
    
    return prompt


def update_prompt_with_observation(
    prompt: List[TextMessage], 
    observation: Dict, 
    observation_type: str
) -> List[TextMessage]:
    """Update prompt with new observation based on observation type."""
    if observation_type == "single":
        # Replace with single latest observation
        return [
            prompt[0], 
            TextMessage(source="assistant", content=f'Previous observations: {observation}')
        ]
    elif observation_type == "all":
        # Append all previous observations
        prompt.append(TextMessage(
            source="assistant", 
            content=f'Previous observations (step-{observation["step"]}): {observation}'
        ))
    elif observation_type == "all_actions":
        # Append all previous actions
        prompt.append(TextMessage(
            source="assistant", 
            content=f'Previous actions (step-{observation["step"]}): {observation["tool"]}'
        ))
    elif observation_type == "null":
        # Don't append any observation
        pass
    
    return prompt


async def process_single_qa(
    qa: Dict, 
    agent: AssistantAgent, 
    args: argparse.Namespace,
    result_dir: str
) -> Optional[Dict]:
    """Process a single QA item with the agent."""
    dp = process_qa(qa, args.explicit_answer)
    dp_path = os.path.join(result_dir, f"{dp['ID']}.json")
    
    # Handle existing files
    if os.path.exists(dp_path) and not args.resume:
        print(f"Overwriting {dp['ID']}")
    
    if args.resume and os.path.exists(dp_path):
        print(f"Resuming {dp['ID']}")
        with open(dp_path, "r", encoding="utf-8") as f:
            dp = json.load(f)
    
    # Create initial prompt
    base_prompt = [TextMessage(
        content=dp["question"] + "\n\n" + thought_prompt, 
        source="user"
    )]
    
    # Set up for resume if needed
    if args.resume and dp.get("cot") and len(dp["cot"]) > 0:
        step = len(dp["cot"]) - 1
        print(f"Resuming from {dp['ID']} at step {step}")
        dp["cot"] = dp["cot"][:-1]  # Remove last incomplete step
        prompt = create_resume_prompt(dp, base_prompt)
    else:
        step = 0
        prompt = base_prompt
    
    # Main processing loop
    while step < MAX_STEPS:
        print(f"Step: {step}")
        try:
            result = await Console(agent.run_stream(
                task=prompt, 
                cancellation_token=CancellationToken()
            ))
        except Exception as e:
            print(f"Error at step {step}: {e}")
            return {
                "id": dp["ID"],
                "prompt": [msg.content for msg in prompt],
                "error": str(e),
                "step": step
            }
        
        step += 1
        messages = result.messages
        
        # Extract information from messages
        thought = None
        tool_call = None
        tool_summary = None
        
        for message in messages:
            if message.type == "ThoughtEvent":
                thought = message.content
            elif message.type == "ToolCallRequestEvent":
                tool_call = message.content[0]
            elif message.type == "ToolCallSummaryMessage":
                tool_summary = message.content
        
        # Handle termination
        if tool_call and "terminate" in tool_call.name.lower():
            observation = {
                "step": step,
                "thought": thought,
                "answer": extract_json(tool_call.arguments)["answer"]
            }
            dp['cot'].append(observation)
            break
        
        # Handle regular tool usage
        else:
            observation = {
                "step": step,
                "thought": thought,
                "tool": {
                    "id": tool_call.id if tool_call else None,
                    "name": tool_call.name if tool_call else None,
                    "arguments": tool_call.arguments if tool_call else None
                },
                "observation": tool_summary
            }
            
            dp['cot'].append(observation)
            prompt = update_prompt_with_observation(prompt, observation, args.observation_type)
    
    # Save results
    with open(dp_path, "w", encoding="utf-8") as f:
        json.dump(dp, f, indent=4)
    
    return None  # No error


async def main() -> None:
    """Main function to orchestrate the QA processing."""
    args = parse_arguments()
    
    # Set up environment
    os.environ["LOG_DIR"] = args.log_dir
    if os.environ.get("IDENTITY") is None:
        os.environ["IDENTITY"] = args.identity
    if os.environ.get("RAG_URL") is None:
        os.environ["RAG_URL"] = rag_url_dict[args.identity]
        assert os.environ["RAG_URL"] is not None, "RAG_URL is not set"
    cache_dir = os.environ.get("CACHE_DIR", args.cache_dir)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load data based on mode
    data_path = os.path.join(
        args.data_path, 
        f"EgoLifeQA_{args.identity}_{identity_mapping_dict[args.identity]}.json"
    )
    
    print(f"Current identity: {args.identity}")
    
    try:
        if args.gen_specific_data:
            print("Loading specific data...")
            # Use specific_data_path or fall back to specific_data_list for compatibility
            specific_path = args.specific_data_path
            if not os.path.exists(specific_path) and hasattr(args, 'specific_data_list'):
                specific_path = args.specific_data_list
            
            if os.path.exists(specific_path):
                target_ids = load_specific_data_ids(specific_path)
            else:
                # Fallback to default path format
                default_path = f"./data_statistics/error_list_{args.identity}.txt"
                target_ids = load_specific_data_ids(default_path)
            
            all_data = load_json_data(data_path)
            egolife_qa_data = filter_data_by_ids(all_data, target_ids)
            print(f"Loaded {len(egolife_qa_data)} specific items")
            
        elif args.resume:
            print("Loading resume data...")
            target_ids = load_resume_data_ids(args.identity)
            all_data = load_json_data(data_path)
            egolife_qa_data = filter_data_by_ids(all_data, target_ids)
            print(f"Loaded {len(egolife_qa_data)} items for resume")
            
        else:
            print(f"Loading all data from: {data_path}")
            egolife_qa_data = load_json_data(data_path)
            print(f"Loaded {len(egolife_qa_data)} items")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if not egolife_qa_data:
        print("No data to process.")
        return
    
    # Set up model and agent components
    model_client = setup_model_client(args)
    tools = get_tools(args)
    sys_prompt_text = get_system_prompt(args)
    
    # Process data
    errors = []
    
    for qa in tqdm(egolife_qa_data, desc="Processing QA items"):
        # Create fresh agent for each QA to avoid state issues
        agent = AssistantAgent(
            name="egolife_qa_agent",
            model_client=model_client,
            tools=tools,
            system_message=sys_prompt_text,
        )
        
        error = await process_single_qa(qa, agent, args, args.result_dir)
        if error:
            errors.append(error)
    
    # Save error list if any
    if errors:
        error_file = os.path.join(args.log_dir, f"error_list_{args.identity}.json")
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=4)
        print(f"Saved {len(errors)} errors to {error_file}")
    else:
        print("No errors encountered!")
    
    # Cleanup
    print("Processing complete!")
    if cache_dir and os.path.exists(cache_dir):
        try:
            os.rmdir(cache_dir)
        except OSError:
            pass  # Directory not empty or doesn't exist
    
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

"""
Data generation package for EgoLifeQA question answering tasks.

This package provides tools and utilities for processing egocentric video data,
generating question-answer pairs, and running AI agents to analyze the data.
"""

# Import main components from modules
from .main import AgentResponse, arg_parser, load_qa_data, main
from .prompts import sys_prompt_v0, sys_prompt_v1, thought_prompt
from .tools import (
    rag, 
    video_llm, 
    vlm, 
    terminate, 
    GPT
)
from .utils import (
    setup_logging_and_config,
    locate_video_url,
    locate_image_url,
    extract_json,
    calculate_time_diff,
    convert_seconds_to_time,
    process_qa
)

# Define what's available when using "from data_gen import *"
__all__ = [
    # From main.py
    'AgentResponse', 'arg_parser', 'load_qa_data', 'main',
    
    # From prompts.py
    'sys_prompt_v0', 'sys_prompt_v1', 'thought_prompt',
    
    # From tools.py
    'rag', 'video_llm', 'vlm', 'terminate', 'GPT',
    
    # From utils.py
    'setup_logging_and_config', 'locate_video_url', 'locate_image_url',
    'extract_json', 'calculate_time_diff', 'convert_seconds_to_time',
    'process_qa'
]

# Package version
__version__ = '0.1.0'

# Package metadata
__author__ = 'Shulin Tian'
__email__ = 'shulin002@e.ntu.edu.sg'
__description__ = 'Tools for generating and processing EgoLifeQA data'

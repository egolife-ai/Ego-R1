import os

RAG_URL = {
    "A1_JAKE": "http://127.0.0.1:8001/query",
    "A2_ALICE": "http://127.0.0.1:8002/query",
    "A3_TASHA": "http://127.0.0.1:8003/query",
    "A4_LUCIA": "http://127.0.0.1:8004/query",
    "A5_KATRINA": "http://127.0.0.1:8005/query",
    "A6_SHURE": "http://127.0.0.1:8006/query",
    "video-mme-long": "http://127.0.0.1:7001/query",
    "egoschema": "http://127.0.0.1:6001/query"
}


# egolife
# VIDEO_LLM_URL = "http://127.0.0.1:8060/video_llm" # api
# VIDEO_LLM_URL = "http://127.0.0.1:8010/video_llm" # llava
# VLM_URL = "http://127.0.0.1:8080/vlm"

# video-mme
# VIDEO_LLM_URL = "http://127.0.0.1:7060/video_llm"
# VLM_URL = "http://127.0.0.1:7090/vlm"

# egoschema
# VIDEO_LLM_URL = "http://127.0.0.1:6060/video_llm"
# VLM_URL = "http://127.0.0.1:6090/vlm"

VIDEO_LLM_URL = os.environ.get("VIDEO_LLM_URL", "http://127.0.0.1:8060/video_llm")
VLM_URL = os.environ.get("VLM_URL", "http://127.0.0.1:8080/vlm")

prompt = """
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

# You should always think before giving an action, i.e., calling a tool or providing the answer. You can call multiple tools in multiple rounds, with each round containing a thinking and a tool call. Your response should only be in one of <think>, <tool>, <answer>, while <information> is provided by the environment. In each round, you must ONLY provide either a tool call or an answer.


format_prompt = """
\n\nIMPORTANT: You should always think before giving an action, i.e., calling a tool or providing the answer. You can call multiple tools in multiple rounds, with each round containing either a thinking with a tool call, or a thinking with an answer. In each round, your response should ONLY be quoted by <think> & <tool>, or <think> & <answer>. DO NOT generate <information> as it should be provided by the environment.
"""
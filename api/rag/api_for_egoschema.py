from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
from r1rag.ragagent import RagAgent
import uvicorn
import json
import argparse

app = FastAPI(title="RAG API for Egoschema", description="API for querying RAG system with video_id")

# 这些变量将在main中通过命令行参数设置
MIN_LOG_DIR = None

class QueryRequest(BaseModel):
    level: str
    keywords: List[str]
    query_time: Optional[str] = None
    start_time: Optional[Union[str, List[Union[str, int]]]] = None
    vid_id: str  # video_id now passed per request

class QueryResponse(BaseModel):
    response: Union[dict, str, List[dict]]

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        # 初始化 RagAgent，每次请求都根据 vid_id 初始化
        min_log = f"{MIN_LOG_DIR}/{request.vid_id}.json"
        print("initializing rag agent")
        rag_agent = RagAgent(day_log=None, hour_log=None, min_log=min_log)
        
        response = rag_agent.query_egoschema(
            level=request.level,
            keywords=request.keywords,
            query_time=request.query_time,
            start_time=request.start_time
        )
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server(host: str = "0.0.0.0", port: int = 6001):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start RAG API for Egoschema with custom min_log dir and port.")
    parser.add_argument("--min_log_dir", type=str, required=True, help="Directory path for egoschema min_log JSON files")
    parser.add_argument("--port", type=int, default=6001, help="Port to run the server on")
    args = parser.parse_args()

    # 设置全局变量
    global MIN_LOG_DIR
    MIN_LOG_DIR = args.min_log_dir

    start_server(port=args.port)

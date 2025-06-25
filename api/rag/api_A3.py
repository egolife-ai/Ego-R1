import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
from r1rag.ragagent import RagAgent
import uvicorn

app = FastAPI(title="RAG API", description="API for querying RAG system at different time levels")

class QueryRequest(BaseModel):
    level: str
    keywords: List[str]
    query_time: Optional[str] = None
    start_time: Optional[Union[str, List[Union[str, int]]]] = None

class QueryResponse(BaseModel):
    response: Union[dict, str, List[dict]]

identity_mapping_dict = {
    "A2": "ALICE",
    "A3": "TASHA",
    "A4": "LUCIA",
    "A5": "KATRINA",
    "A6": "SHURE"
}

global rag_agent
rag_agent = None  # Will be initialized in main

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system at different time levels (week, day, hour)
    """
    try:
        response = rag_agent.query(
            level=request.level,
            keywords=request.keywords,
            query_time=request.query_time,
            start_time=request.start_time
        )
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def start_server(host: str = "0.0.0.0", port: int = 8003):
    """
    Start the FastAPI server
    """
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start RAG API server with custom JSON logs and port.")
    parser.add_argument("--day_log", type=str, required=True, help="Path to day log JSON file")
    parser.add_argument("--hour_log", type=str, required=True, help="Path to hour log JSON file")
    parser.add_argument("--min_log", type=str, required=True, help="Path to min log JSON file")
    parser.add_argument("--port", type=int, default=8003, help="Port to run the server on")
    args = parser.parse_args()

    # Initialize RAG agent with command line arguments
    rag_agent = RagAgent(day_log=args.day_log, hour_log=args.hour_log, min_log=args.min_log)

    start_server(port=args.port)
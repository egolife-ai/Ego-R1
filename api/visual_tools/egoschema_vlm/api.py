from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from vlm import vlm
import os
import yaml
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config()
DATA_DIR = config['egoschema']['vlm']['data_dir']
VLM_PORT = config['egoschema']['vlm']['vlm_port']
app = FastAPI(title="Vision Language Model API")

class VLMRequest(BaseModel):
    question: str
    timestamp: str
    vid_id:str

class VLMResponse(BaseModel):
    query_time: str
    question: str
    answer: str
    exact_match: bool

@app.post("/vlm", response_model=VLMResponse)
async def vlm_endpoint(request: VLMRequest):
    """
    Analyze a single image frame at the specified timestamp using a vision language model.
    
    Parameters:
    - question: The question to ask about the image
    - timestamp: The timestamp in format DAYX_HHMMSSFF (X is day number, HHMMSS is hour:minute:second, FF is frame number 00-19)
    
    Returns:
    - query_time: The timestamp used
    - question: The question asked
    - answer: The model's response
    - exact_match: Whether the image was found exactly
    
    """
    print(request)
    try:
        data_dir = f"{DATA_DIR}/{request.vid_id}"
        result = vlm(question=request.question, timestamp=request.timestamp, data_dir=data_dir)
        return result
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=VLM_PORT)

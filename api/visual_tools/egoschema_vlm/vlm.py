from utils import *

def vlm(question: Annotated[str, "The question you want to use the vision language model to answer. Note: The question does not necessarily need to be the same as the question given by the user. You should propose a question based on the available observations."], timestamp: Annotated[str, "The timestamp of the video to answer the question. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))"], data_dir: Annotated[str, "The directory of all videos"] = "/home/data2/sltian/code/Ego-R1_dev/EgoLife") -> dict:
    """
    Analyze a single image frame at the specified timestamp using a vision language model.
    Answers questions about visual scenes, events, and objects.
    Returns both the answer and image file metadata.
    
    FREQUENCY: 2/10    
    COST: 2/5
    TEMPORAL CAPABILITY: 1/5
    VISUAL INPUT: True
    
    ## HINT
    1. Ask targeted questions that align with your observation and reasoning process rather than repeating the user's original question.
    """
    
    # Use OpenAI API to answer the question
    os.makedirs("./cache", exist_ok=True)
    frame_path, exact_match = locate_image_url(timestamp,data_dir)
    print(f"Frame path: {frame_path}")
    image_client = GPT()
    answer = image_client.chat(question, frame_path)
    
    response = {
        "query_time": timestamp,
        "question": question,
        "answer": answer,
        "exact_match": exact_match
    }
    return response
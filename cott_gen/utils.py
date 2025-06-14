import os
import logging
from datetime import datetime
from typing_extensions import Annotated
from autogen_core import TRACE_LOGGER_NAME, EVENT_LOGGER_NAME
import cv2
import os.path as osp
import random
import json
import ast
from openai import AzureOpenAI
import base64
import ffmpeg

# TODO: add the model config here or in the bash file; gemini api can be used as video_llm model; gpt-4o can serve as the vlm model; gpt-4.1 can be used as llm agent backbone
model_config = {
    "Gemini_API_KEY": os.getenv("GEMINI_API_KEY", ""),
    "gpt-4o": {
        "endpoint": os.getenv("ENDPOINT_URL", ""),
        "deployment": os.getenv("DEPLOYMENT_NAME", "gpt-4o"),
        "subscription_key": os.getenv("AZURE_OPENAI_API_KEY", "")
    },
    "gpt-4.1": {
        "endpoint": os.getenv("ENDPOINT_URL", ""),
        "deployment": os.getenv("DEPLOYMENT_NAME", "gpt-4.1"),
        "subscription_key": os.getenv("AZURE_OPENAI_API_KEY", "")
    }
    
}

identity_mapping_dict = {
    "A1": "JAKE",
    "A2": "ALICE",
    "A3": "TASHA",
    "A4": "LUCIA",
    "A5": "KATRINA",
    "A6": "SHURE"
}

rag_url_dict = {
    "A1": "http://localhost:8001/query",
    "A2": "http://localhost:8002/query",
    "A3": "http://localhost:8003/query",
    "A4": "http://localhost:8004/query",
    "A5": "http://localhost:8005/query",
    "A6": "http://localhost:8006/query"
}

def setup_logging_and_config(model: str):
    # Set up logging
    LOG_DIR = os.environ["LOG_DIR"]
    os.makedirs(LOG_DIR, exist_ok=True)

    log_file = f"{LOG_DIR}/ego-r1_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.environ["LOG_FILE"] = log_file
    print(f"Logs will be saved to: {os.path.abspath(log_file)}")
    # Configure handlers
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]

    # Set up loggers
    for logger_name in [TRACE_LOGGER_NAME, EVENT_LOGGER_NAME]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        for handler in handlers:
            logger.addHandler(handler)

    if "4o" in model:
        return model_config["gpt-4o"].values()
    elif "4.1" in model:
        return model_config["gpt-4.1"].values()
    else:
        raise ValueError(f"Invalid model: {model}")

class GPT():
    def __init__(self, sys_prompt="You are a helpful assistant.", model: str = "gpt-4o"):
        self.endpoint, self.deployment, self.subscription_key = model_config[model].values()
        self.client = AzureOpenAI(  
            azure_endpoint=self.endpoint,  
            api_key=self.subscription_key,  
            api_version="2024-05-01-preview",  
        )
        self.sys_prompt = sys_prompt

    def chat(self, message, image_path=None):
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.sys_prompt,
                    }
                ]
            },
            {
                "role": "user",
                "content": message
            }
        ]

        if image_path:
            encoded_image = self.encode_image(image_path)
            
            chat_prompt.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}" # f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }   
            )
        
        # Include speech result if speech is enabled  
        messages = chat_prompt  
            
        # Generate the completion  
        completion = self.client.chat.completions.create(  
            model=self.deployment,  
            messages=messages,  
            max_tokens=800,  
            temperature=0.7,  
            top_p=0.95,  
            frequency_penalty=0,  
            presence_penalty=0,  
            stop=None,  
            stream=False
        )
        result = completion.choices[0].message.content
        # print(completion.to_json())
        return result
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            # return base64.b64encode(image_file.read()).decode('utf-8')  # GPT-4Vision
            return base64.b64encode(image_file.read()).decode('ascii')  # Azure 


    
def llm_fuzzy_match(question: str, options: str, answer: str) -> bool:
    """Fuzzy match the question and answer"""
    client = GPT()
    prompt = f"The question is: {question}\nThe options available are: {options}\nThe free-form answer is: {answer}\nPlease determine if the answer is in the options. If yes, please return the answer quoted by ``````,. For example, ```A```. Otherwise, return ```NO```."
    response = client.chat(prompt)
    print(response)
    return response

def trim_video(input_file, output_file, trim_info):
    """
    Trim a video based on the position (first, last, or middle)
    
    Args:
        input_file (str): Path to input video
        output_file (str): Path to save trimmed video
        trim_info (dict): Dictionary containing trim information
            - For first video: {'position': 'first', 'start_s': start_time_in_seconds}
            - For last video: {'position': 'last', 'end_s': end_time_in_seconds}
            - For middle videos: {'position': 'middle', 'start_s': start_time, 'end_s': end_time}
    """
    import ffmpeg
    
    position = trim_info.get('position', 'middle')
    
    if position == 'first':
        # For first video: trim from start_s to the end
        start_time = trim_info['start_s']
        (
            ffmpeg
            .input(input_file, ss=start_time)
            .output(output_file, c='copy')
            .run()
        )
    elif position == 'last':
        # For last video: trim from beginning to end_s
        end_time = trim_info['end_s']
        (
            ffmpeg
            .input(input_file, to=end_time)
            .output(output_file, c='copy')
            .run()
        )
    else:
        # For middle videos: trim from start_s to end_s
        start_time = trim_info['start_s']
        end_time = trim_info['end_s']
        (
            ffmpeg
            .input(input_file, ss=start_time, to=end_time)
            .output(output_file, c='copy')
            .run()
        )
    
    return output_file

def config_videos(file_paths, start_timestamp, end_timestamp):
    file_s = file_paths[0]
    file_e = file_paths[-1]
    
    # get the duration of the videos
    file_s_duration = float(ffmpeg.probe(file_s)["streams"][0]["duration"]) # 30.000000 -> 30.0
    file_e_duration = float(ffmpeg.probe(file_e)["streams"][0]["duration"]) # 30.000000 -> 30.0
    
    start_time = start_timestamp.split("_")[1]
    end_time = end_timestamp.split("_")[1]
    
    file_s_start = file_s.split("_")[-1].split(".")[0]
    file_e_start = file_e.split("_")[-1].split(".")[0]  # Changed from file_e_end to file_e_start
    
    # Calculate trim points
    # For first file: trim from (start_time - file_s_start) to end
    # For last file: trim from start to (end_time - file_e_start)
    trim_s = max(0, time_to_seconds(start_time) - time_to_seconds(file_s_start))
    trim_e = min(file_e_duration, time_to_seconds(end_time) - time_to_seconds(file_e_start))
    
    if trim_e < 0:
        raise ValueError("No video information in this time range. Try other tools or other time range.")
    # trim the video
    if len(file_paths) == 1:
        file_s_trim = trim_video(
            file_s, 
            os.path.join(os.environ["CACHE_DIR"], "processed_video.mp4"), 
            {'position': 'middle', 'start_s': trim_s, 'end_s': trim_e}
        )
        
        length = float(ffmpeg.probe(file_s_trim)["streams"][0]["duration"])
        return file_s_trim, length
    
    else:
        file_s_trim = trim_video(
            file_s, 
            os.path.join(os.environ["CACHE_DIR"], file_s.split("/")[-1]), 
            {'position': 'first', 'start_s': trim_s}
        )
        
        if trim_e == 0:
            file_paths.pop()
        else:
            file_e_trim = trim_video(
                file_e, 
                os.path.join(os.environ["CACHE_DIR"], file_e.split("/")[-1]), 
                {'position': 'last', 'end_s': trim_e}
            )

        # Create a temporary file list for ffmpeg
        with open(os.path.join(os.environ["CACHE_DIR"], "temp_file_list.txt"), "w") as f:
            f.write(f"file '{os.path.abspath(file_s_trim)}'\n")
            for file in file_paths[1:-1]:
                f.write(f"file '{os.path.abspath(file)}'\n")
            try:
                f.write(f"file '{os.path.abspath(file_e_trim)}'\n")
            except:
                pass
            
        # combine the trimmed videos
        combined_video = os.path.join(os.environ["CACHE_DIR"], "processed_video.mp4")
        (
            ffmpeg
            .input(os.path.join(os.environ["CACHE_DIR"], "temp_file_list.txt"), format='concat', safe=0)
            .output(combined_video, c='copy')
            .run()
        )
        length = float(ffmpeg.probe(combined_video)["streams"][0]["duration"])
        # delete the temporary file list
        os.remove(os.path.join(os.environ["CACHE_DIR"], "temp_file_list.txt"))
        return combined_video, length
    
    
    
def locate_videos(start_timestamp: Annotated[str, "The start timestamp of the video to answer the question. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))"], end_timestamp: Annotated[str, "The end timestamp of the video to answer the question. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))"], data_dir: Annotated[str, "The directory of all videos"] = "./EgoLife", identity: Annotated[str, "The identity of the person in the video"] = "A1") -> str:
    """Locate the video url based on the timestamp. The video is stored in the data_dir. We should return the video file path. Usually, the video name is DAYx_AN_NAME_HHMMSSFF.mp4"""
    
    start_day, start_time = start_timestamp.split("_")
    end_day, end_time = end_timestamp.split("_")
    assert start_day == end_day, f"The start and end day are not the same: {start_day} != {end_day}"
    assert start_time <= end_time, f"The start time is greater than the end time: {start_time} > {end_time}"
    
    # if the input is greater that 10 minutes, we should sample the video at 10 fps
    if abs(calculate_time_diff(end_timestamp, start_timestamp)) > 600:
        raise ValueError(f"The time span is greater than 10 minutes: {start_timestamp} to {end_timestamp}")
    else:
        start_url, start_exact_match = locate_video_url(timestamp=start_timestamp, identity=identity)
        end_url, end_exact_match = locate_video_url(timestamp=end_timestamp, identity=identity)
        
        if start_url == end_url:
            return [start_url], start_exact_match and end_exact_match
        else:
            # Get video filenames and directory path
            video_dir = os.path.dirname(start_url)
            start_video = os.path.basename(start_url)
            end_video = os.path.basename(end_url)
            
            # Get all videos in chronological order
            videos = sorted(os.listdir(video_dir))
            
            # Find the range of videos between start and end timestamps
            start_idx = videos.index(start_video)
            end_idx = videos.index(end_video) + 1  # Include the end video
            
            # Create list of full paths to selected videos
            selected_videos = [os.path.join(video_dir, video) for video in videos[start_idx:end_idx]]
            return selected_videos, start_exact_match and end_exact_match



def locate_video_url(timestamp: Annotated[str, "The timestamp of the video to answer the question. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))"], data_dir: Annotated[str, "The directory of all videos"] = "./EgoLife", identity: Annotated[str, "The identity of the person in the video"] = "A1") -> str:
    """Locate the video url based on the timestamp. The video is stored in the data_dir. We should return the video file path. Usually, the video name is DAYx_AN_NAME_HHMMSSFF.mp4"""
    
    exact_match = False
    try:
        if not timestamp.startswith("DAY") or "_" not in timestamp:
            raise ValueError(f"Invalid timestamp: {timestamp}")
        day_part, time_part = timestamp.split("_")
        day = day_part[3:]

        if not day.isdigit() or int(day) > 7 or int(day) < 1:
            raise ValueError(f"Invalid day: {day}")
        
        hh = time_part[0:2]
        mm = time_part[2:4]
        ss = time_part[4:6]
        ff = time_part[6:8]
        
        if len(time_part) < 6:  # Need at least HHMMSS
            raise ValueError(f"Invalid time format in timestamp: {time_part}. Expected at least HHMMSS")
        
        hh = time_part[:2]
        mm = time_part[2:4]
        ss = time_part[4:6]
        ff = time_part[6:8] if len(time_part) >= 8 else "00"  # Default to 00 if frame not provided
        
        if int(ff) > 19:
            ff = "00"
        # Convert timestamp to seconds for comparison
        target_seconds = int(hh) * 3600 + int(mm) * 60 + int(ss)
        
        # load the daily video list
        day_dir = os.path.join(data_dir, f"{identity}_{identity_mapping_dict[identity]}", f"DAY{day}")
        daily_videos = os.listdir(day_dir)
        
        # Find best matching video
        best_video = None
        min_diff = float('inf')
        
        for video in daily_videos:
            if not video.endswith(".mp4"):
                continue
            time_str = video.split("_")[-1].split(".")[0]
            v_hh, v_mm, v_ss, v_ff = time_str[:2], time_str[2:4], time_str[4:6], time_str[6:8]
            video_seconds = int(v_hh) * 3600 + int(v_mm) * 60 + int(v_ss)
            
            # video_length = get_video_length_cv2(os.path.join(day_dir, video))
            # Check if timestamp falls within this 30-second video
            if video_seconds <= target_seconds < video_seconds + 30:
                exact_match = True
                return os.path.join(day_dir, video), exact_match

            # track the closest video
            diff = abs(target_seconds - video_seconds)
            if diff < min_diff:
                min_diff = diff
                best_video = video
        
    except Exception as e:
        logging.error(f"Error locating video url: {str(e)}")
        raise e
        
    if best_video is not None:
        return os.path.join(day_dir, best_video), exact_match
    
    raise ValueError(f"No video file found for the timestamp {timestamp}")


def locate_image_url(timestamp: Annotated[str, "The timestamp of the image to answer the question. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))"], data_dir: Annotated[str, "The directory of all images"] = "./EgoLife", identity: Annotated[str, "The identity of the person in the video"] = "A1") -> str:
    """Locate the image url based on the timestamp. Get a screenshot from the video based on the timestamp."""
    video_path, exact_match = locate_video_url(timestamp=timestamp, identity=identity)
    ff = timestamp[6:8]
    # if the proposed frame number is greater than 19, set it to the first frame
    if int(ff) >= 20:
        ff = '00'
    if exact_match:
        # Get the screenshot from the video
        video_cap = cv2.VideoCapture(video_path)
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, int(ff))
        ret, frame = video_cap.read()
        video_cap.release()
        
    else:
        # Get the screenshot from the video of a random frame
        video_cap = cv2.VideoCapture(video_path)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        random_frame = random.randint(0, frame_count - 1)
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
        ret, frame = video_cap.read()
        video_cap.release()
        
    # Save the frame to the cache
    image_path = os.path.join(os.environ["CACHE_DIR"], f"{timestamp}.png")
    cv2.imwrite(image_path, frame)
    return image_path, exact_match

def sample_fps(video_path: str, fps: int = 10) -> str:
    """Sample the video at the given fps"""
    video_cap = cv2.VideoCapture(video_path)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = frame_count // fps
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, sample_interval)
    ret, frame = video_cap.read()
    video_cap.release()

def extract_json(s: str) -> dict:
    """Extract the json from the string"""
    # find the first { and the last }
    
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            # If that fails, try to parse as Python literal
            return ast.literal_eval(s)
        except (SyntaxError, ValueError):
            # If both fail, try to clean up the string
            # Replace single quotes with double quotes
            s = s.replace("'", '"')
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse string as JSON: {s}")


def time_to_seconds(time: str) -> int:
    """Convert the timestamp HHMMSS(FF) to the seconds"""
    hh, mm, ss = time[:2], time[2:4], time[4:6]
    return int(hh) * 3600 + int(mm) * 60 + int(ss)

def calculate_time_diff(timestamp_1: str, timestamp_2: str) -> int:
    """Calculate the time difference between two timestamps. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))"""
    print("The query timestamp is: ", timestamp_1, "The target timestamp is: ", timestamp_2)
    day_1, time_1 = timestamp_1.split("_")
    day_2, time_2 = timestamp_2.split("_")
    day_1 = int(day_1[3:])
    day_2 = int(day_2[3:])
    seconds_1 = day_1 * 24 * 3600 + time_to_seconds(time_1)
    seconds_2 = day_2 * 24 * 3600 + time_to_seconds(time_2)
    
    return seconds_1 - seconds_2

def convert_seconds_to_time(seconds: int) -> str:
    """Convert the seconds to the timestamp format"""
    day = seconds // (24 * 3600)
    seconds = seconds % (24 * 3600)
    hh = seconds // 3600
    seconds = seconds % 3600
    mm = seconds // 60
    ss = seconds % 60
    return f"{day} day {hh:02d}:{mm:02d}:{ss:02d}"

def process_qa(qa: dict, explicit_answer: bool = False) -> dict:
    """Process the qa data"""
    
    result = {}
    id = qa["ID"]
    question = qa["question"]
    query_timestamp = qa["query_time"]["date"] + "_" + qa["query_time"]["time"]
    
    # get the last key-value pair of the target_time
    target_timestamp = ""  
    k, v = list(qa["target_time"].items())[-1]
    if k == "time" and isinstance(v, str):
        target_timestamp = qa["target_time"]["date"] + "_" + v[:8]
    elif isinstance(v, list):
        target_timestamp = qa["target_time"]["date"] + "_" + v[-1][:8]
    else:
        raise ValueError(f"Invalid target_time: {qa['target_time']}")
    options = f"A. {qa['choice_a']}\nB. {qa['choice_b']}\nC. {qa['choice_c']}\nD. {qa['choice_d']}"
    result["ID"] = id
    if explicit_answer:
        result["question"] = question + " <timestamp>" + query_timestamp + "</timestamp>\n"
        result["options"] = options
    else:
        result["question"] = question + " <timestamp>" + query_timestamp + "</timestamp>\n" + options
    result['cot'] = []
    result["ground_truth"] = qa["answer"]
    result["reasoning_span"] = {
        "query_time": query_timestamp,
        "target_time": target_timestamp,
        "span": calculate_time_diff(query_timestamp, target_timestamp),
        "span_str": convert_seconds_to_time(calculate_time_diff(query_timestamp, target_timestamp))
    }
    result["log"] = os.environ["LOG_FILE"]
    return result


# video_url, exact_match = locate_video_url("DAY1_11200000")
# print(video_url, exact_match)


# print(locate_videos("DAY1_11000428", "DAY1_11100430"))
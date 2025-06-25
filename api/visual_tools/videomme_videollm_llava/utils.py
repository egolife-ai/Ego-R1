import os
import logging
from datetime import datetime
from typing_extensions import Annotated
import cv2
import os.path as osp
import random
import json
import ast
from openai import AzureOpenAI
import base64
import ffmpeg
from typing import List, Tuple


identity_mapping_dict ={
    "A1": "JAKE",
    "A2": "ALICE",
    "A3": "TASHA",
    "A4": "LUCIA",
    "A5": "KATRINA",
    "A6": "SHURE"
}



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
            .input(input_file)
            .output(output_file, c='copy', to=end_time)
            .run()
        )
    else:
        # For middle videos: trim from start_s to end_s
        start_time = trim_info['start_s']
        end_time = trim_info['end_s']
        (
            ffmpeg
            .input(input_file, ss=start_time)
            .output(output_file, to=end_time, c='copy')
            .run()
        )
    return output_file
def downsample_video(input_path: str, output_path: str, fps: int = 1, resize: bool = True) -> None:
    """Uniformly sample *input_path* at *fps* and optionally resize to 448×448, saving to *output_path*."""
    vf = f"fps={fps}"
    if resize:
        vf += ",scale=448:448"
    (
        ffmpeg
        .input(input_path)
        .output(
            output_path,
            vf=vf,
            vcodec="libx264",
            pix_fmt="yuv420p",
            r=fps,  # ensure container reports correct fps
            loglevel="error",
        )
        .overwrite_output()
        .run()
    )

def config_videos(
    file_paths: List[str],
    start_timestamp: str,
    end_timestamp: str,
    cache_dir: str | None = None,
    downsample_fps: int = 1,
    resize: bool = True,
) -> Tuple[str, float]:
    """Return a stitched + down‑sampled video (1 fps, 448×448) covering *start*→*end*.

    Parameters
    ----------
    file_paths : list[str]
        Chronologically ordered raw clip paths covering the target period.
    start_timestamp, end_timestamp : str
        "DAYX_HHMMSSFF" strings marking target span (inclusive).
    cache_dir : str | None
        Directory for temporary files & output.
    downsample_fps : int, default 1
        Target output frame‑rate.
    resize : bool, default True
        Whether to resize frames to 448 × 448; set *False* for maximum speed.
    """
    if cache_dir is None:
        cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)

    file_s, file_e = file_paths[0], file_paths[-1]
    file_s_start = os.path.basename(file_s).split("_")[-1].split(".")[0]
    file_e_start = os.path.basename(file_e).split("_")[-1].split(".")[0]

    file_s_duration = float(ffmpeg.probe(file_s)["streams"][0]["duration"])
    file_e_duration = float(ffmpeg.probe(file_e)["streams"][0]["duration"])

    start_time = start_timestamp.split("_")[1]
    end_time = end_timestamp.split("_")[1]

    trim_s = max(0, time_to_seconds(start_time) - time_to_seconds(file_s_start))
    trim_e = min(file_e_duration, time_to_seconds(end_time) - time_to_seconds(file_e_start))

    if trim_e < 0:
        raise ValueError("No video information in this time range. Try a different span.")

    # ------------------------------------------------------------------
    # 1. Trim individual clips and concatenate (lossless copy) ---------
    # ------------------------------------------------------------------
    concat_video = os.path.join(cache_dir, "concat_tmp.mp4")

    if len(file_paths) == 1:
        # Single clip: cut middle portion in one go
        trim_video(
            file_s,
            concat_video,
            {"position": "middle", "start_s": trim_s, "end_s": trim_e},
        )
    else:
        # First & last clips get trimmed; middle clips copied as‑is
        first_trim = trim_video(
            file_s,
            os.path.join(cache_dir, f"trim_0_{os.path.basename(file_s)}"),
            {"position": "first", "start_s": trim_s},
        )
        clips_to_merge = [first_trim]

        # middle untouched clips
        if len(file_paths) > 2:
            clips_to_merge.extend(file_paths[1:-1])

        # possible last trim
        if trim_e > 0:
            last_trim = trim_video(
                file_e,
                os.path.join(cache_dir, f"trim_n_{os.path.basename(file_e)}"),
                {"position": "last", "end_s": trim_e},
            )
            clips_to_merge.append(last_trim)
        else:
            clips_to_merge.append(file_e)

        # create ffmpeg concat list
        list_path = os.path.join(cache_dir, "file_list.txt")
        with open(list_path, "w") as list_file:
            for clip in clips_to_merge:
                list_file.write(f"file '{os.path.abspath(clip)}'\n")

        (
            ffmpeg
            .input(list_path, format="concat", safe=0)
            .output(concat_video, c="copy", loglevel="error")
            .overwrite_output()
            .run()
        )
        os.remove(list_path)

    # ------------------------------------------------------------------
    # 2. Down‑sample (+optional resize) --------------------------------
    # ------------------------------------------------------------------
    processed_video = os.path.join(cache_dir, "processed_video.mp4")
    downsample_video(concat_video, processed_video, fps=downsample_fps, resize=resize)

    # gather duration metadata
    length = float(ffmpeg.probe(processed_video)["streams"][0]["duration"])

    # clean up temporary concat file
    if os.path.exists(concat_video):
        os.remove(concat_video)

    return processed_video, length


def locate_videos(start_timestamp, end_timestamp, data_dir="./EgoLife", cache_dir=None):
    video_list=sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.mp4')])
    clip_start_time=video_list[0].split(".")[0]
    clip_end_time=video_list[-1].split(".")[0]
    # Take last 6 digits and pad with leading "00"
    start_time = "00" + start_timestamp.split("_")[-1][-6:]
    end_time = "00" + end_timestamp.split("_")[-1][-6:]
    
    # Convert times to minutes for comparison
    start_minutes = int(start_time[:2])*60 + int(start_time[2:4])
    end_minutes = int(end_time[:2])*60 + int(end_time[2:4])
    
    assert end_minutes - start_minutes <= 10, "Time range exceeds 10 minutes"
    assert start_time <= end_time, f"The start time is greater than the end time: {start_time} > {end_time}"
    
    print(start_time, end_time)
    if start_time > clip_end_time:
        print(f"The start time is greater than the end time: {start_time} > {clip_end_time}")
        # 将end_time设为clip_end_time
        end_time = clip_end_time
        # 将start_time设为clip_end_time减去10分钟，保持HHMMSSFF格式
        hh = int(clip_end_time[:2])
        mm = int(clip_end_time[2:4])
        ss = int(clip_end_time[4:6])
        ff = clip_end_time[6:8]
        # 计算clip_end_time的总分钟数
        total_minutes = hh * 60 + mm
        # 减去10分钟
        new_total_minutes = max(0, total_minutes - 10)
        new_hh = new_total_minutes // 60
        new_mm = new_total_minutes % 60
        # 保持秒和帧不变，拼接成8位
        start_time = f"{new_hh:02d}{new_mm:02d}{ss:02d}{ff}"
        
        # 继续后续逻辑
    
    start_timestamp=f"DAY1_{start_time}"
    end_timestamp=f"DAY1_{end_time}"
    start_url, start_exact_match = locate_video_url(timestamp=start_timestamp, data_dir=data_dir)
    end_url, end_exact_match = locate_video_url(timestamp=end_timestamp, data_dir=data_dir)
    if start_url == end_url:
        return [start_url], start_exact_match and end_exact_match,start_timestamp
    else:
        video_dir = os.path.dirname(start_url)
        start_video = os.path.basename(start_url)
        end_video = os.path.basename(end_url)
        videos = sorted(os.listdir(video_dir))
        start_idx = videos.index(start_video)
        end_idx = videos.index(end_video) + 1
        selected_videos = [os.path.join(video_dir, video) for video in videos[start_idx:end_idx]]
        return selected_videos, start_exact_match and end_exact_match, start_timestamp
        
def locate_video_url(timestamp: Annotated[str, "The timestamp of the video to answer the question. The format should be DAYX_HHMMSSFF (X is the day number, HHMMSS is the hour, minute, second, and FF is the frame number(00~19))"], data_dir: Annotated[str, "The directory of all videos"] = "/home/data2/sltian/code/Ego-R1_dev/EgoLife") -> str:
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
        
        all_videos = os.listdir(data_dir)
        
        # Find best matching video
        best_video = None
        min_diff = float('inf')
        
        for video in all_videos:
            if not video.endswith(".mp4"):
                continue
            time_str = video.split("_")[-1].split(".")[0]
            v_hh, v_mm, v_ss, v_ff = time_str[:2], time_str[2:4], time_str[4:6], time_str[6:8]
            video_seconds = int(v_hh) * 3600 + int(v_mm) * 60 + int(v_ss)
            
            # video_length = get_video_length_cv2(os.path.join(day_dir, video))
            # Check if timestamp falls within this 30-second video
            if video_seconds <= target_seconds < video_seconds + 30:
                exact_match = True
                return os.path.join(data_dir, video), exact_match

            # track the closest video
            diff = abs(target_seconds - video_seconds)
            if diff < min_diff:
                min_diff = diff
                best_video = video
        
    except Exception as e:
        logging.error(f"Error locating video url: {str(e)}")
        raise e
        
    if best_video is not None:
        return os.path.join(data_dir, best_video), exact_match
    
    raise ValueError(f"No video file found for the timestamp {timestamp}")


def locate_image_url(timestamp, data_dir="/home/data2/sltian/code/Ego-R1_dev/EgoLife", identity="A1", cache_dir=None):
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
    if cache_dir is None:
        cache_dir = "./cache"
    image_path = os.path.join(cache_dir, f"{timestamp}.png")
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


def get_total_duration_from_videos(data_dir: str) -> int:
    """
    Estimate the total duration of all video clips based on filename patterns (HHMMSSFF).
    Assumes each video is a 30-second segment and filenames are in HHMMSSFF.mp4 format.
    Returns duration in seconds.
    """
    video_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".mp4")])
    if not video_files:
        return 0

    # Get the last video filename
    last_video = video_files[-1]
    base_time_str = last_video.split("_")[-1].split(".")[0]  # HHMMSSFF
    hh, mm, ss = int(base_time_str[:2]), int(base_time_str[2:4]), int(base_time_str[4:6])
    base_seconds = hh * 3600 + mm * 60 + ss

    # Add 30 seconds because the filename indicates the start time of that segment
    return base_seconds 
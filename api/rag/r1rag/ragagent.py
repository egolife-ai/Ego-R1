from abc import ABC, abstractmethod
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Callable, Union
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import os
import pickle
from r1rag.utils import *
from r1rag.prompts import *
from tqdm import tqdm
import re
import codecs
class RagAgent(ABC):
    def __init__(self, day_log=None,hour_log=None,min_log=None):
        """
        Initialize RagAgent with a custom embedding model.
        
        """
        
        if day_log is not None:
            with open(day_log, 'r') as f:
                day_log_data=json.load(f)
            self.day_log=day_log_data
        else:
            self.day_log=None
        if hour_log is not None:
            with open(hour_log, 'r') as f:
                hour_log_data=json.load(f)
            self.hour_log=hour_log_data
        else:
            self.hour_log=None
        if min_log is not None:
            with open(min_log, 'r') as f:
                min_log_data=json.load(f)
            self.min_log=min_log_data
        else:
            self.min_log=None
            
            

    def query_week(self,current_time,keywords):
        """
        Query the database for relevant embeddings based on keywords and current time.
        
        Args:
            current_time: Current time in HHMMSSFF format
            keywords: List of keywords to search for
        """
        start_time = time.time()
        
        if current_time is not None:
            # Extract day number and time from current_time format (DAYX_XXXXXXXX)
            date = int(current_time[3]) # Extract X from DAYX
            time_val = current_time.split('_')[1] # Extract XXXXXXXX after _
        else:
            date = 7
            time_val = 24000000
        
        if self.day_log is None:
            return "No day log data available"
        
        day_data=[
            entry for entry in self.day_log
            if entry['date'] <= date
        ]
        
       
        prompt = f"Day summaries for all relevant dates: {day_data} \n keywords: {keywords}"
        response_raw = call_gpt4(system_message=day_prompt, prompt=prompt, max_tokens=1000)
        print(response_raw)
        # Try up to 3 times to get properly formatted response
        attempts = 0
        while attempts < 3:
            if not response_raw.startswith('```') and not response_raw.endswith('```') and not response_raw.startswith('['):
                response_raw = call_gpt4(system_message=day_prompt, prompt=prompt, max_tokens=1000)
                attempts += 1
            else:
                break
                
        # If all attempts failed, return entries for current date
        if attempts == 3:
            return [entry for entry in day_data if entry['date'] == date]
        response = strip_code_fences(response_raw)
        response = json.loads(strip_code_fences(response_raw))
        try:
            first=response[0]
            first['level']='week' 
            first['exact_match']=True
        except:
            first=day_data[-1]
            if 'generated_text' in first:
                first['relevant_content'] = first.pop('generated_text')
            first['exact_match']=False
            first['level']='week'
        return first
    
    def query_day(self,date,keywords):
        """
        Query the database for relevant embeddings based on keywords and day range.
        
        Args:
            day_range: List of day numbers to search for
            keywords: List of keywords to search for
        """
        
        hour_data=[
            entry for entry in self.hour_log
            if entry['date'] == date
        ]
        
        prompt=f"Day summaries {hour_data} \n keywords {keywords}"
        response_raw = call_gpt4(system_message=hour_prompt, prompt=prompt, max_tokens=1000)
        
        attempts = 0
        while attempts < 3:
            if not response_raw.startswith('```') and not response_raw.endswith('```') and not response_raw.startswith('['):
                response_raw = call_gpt4(system_message=day_prompt, prompt=prompt, max_tokens=1000)
                attempts += 1
            else:
                break
                
        # If all attempts failed, return entries for current date
        if attempts == 3:
            return hour_data[0]
        response = json.loads(strip_code_fences(response_raw))
        try:
            first=response[0]
            first['level']='day'  
            first['exact_match']=True
        except:
            first=hour_data[-1]
            if 'generated_text' in first:
                first['relevant_content'] = first.pop('generated_text')
            first['exact_match']=False
            first['level']='day'
        return first
    
    def query_hour(self, start_time, keywords):
        
        # Extract day number and time from current_time format (DAYX_XXXXXXXX)
        date = int(start_time[3]) # Extract X from DAYX
        log_start_time = int(start_time.split('_')[1]) # Extract XXXXXXXX after _
        log_end_time = log_start_time + 1000000
        min_data=[
            entry for entry in self.min_log
            if entry['date'] == date and entry['start_time'] >= log_start_time and entry['end_time'] <= log_end_time
        ]
        if len(min_data)==0:
            # If no data found in the exact hour range, find the closest hour data
            hour_data = [
                entry for entry in self.hour_log 
                if entry['date'] == date
            ]
            if len(hour_data) == 0:
                return "No available data for Hour level Query"
            
            # Find closest hour entry by comparing start times
            closest = min(hour_data, key=lambda x: abs(x['start_time'] - log_start_time))
            min_data = [closest]
        prompt=f"Hour summaries {min_data} \n keywords {keywords}"
        response_raw = call_gpt4(system_message=min_prompt, prompt=prompt, max_tokens=800)
        response = strip_code_fences(response_raw)
        response = json.loads(strip_code_fences(response_raw))
        try:
            first=response[0]
            first['level']='hour'       
            first['exact_match']=True
        except:
            first=min_data[-1]
            if 'generated_text' in first:
                first['relevant_content'] = first.pop('generated_text')
            first['exact_match']=False
            first['level']='hour'
        return first
    
    def query(
        self,
        level,
        keywords,
        query_time=None,
        start_time=None,
    ):
        # Check if query_time and start_time are provided
        if query_time and start_time:
            # Handle list of start times
            if isinstance(start_time, list):
                for st in start_time:
                    # Extract day and timestamp
                    query_day = int(query_time[3])
                    query_ts = int(query_time.split('_')[1])
                    start_day = int(st[3])
                    start_ts = int(st.split('_')[1])
                    if start_day > query_day or (start_day == query_day and start_ts > query_ts):
                        return "error: invalid search. start_time should not later than query_time"
            else:
                # Extract day and timestamp
                query_day = int(query_time[3])
                query_ts = int(query_time.split('_')[1])
                start_day = int(start_time[3])
                start_ts = int(start_time.split('_')[1])
                if start_day > query_day or (start_day == query_day and start_ts > query_ts):
                    return "error: invalid search. start_time should not later than query_time"
        if level == "week":
            return self.query_week(query_time, keywords)
        elif level == "day":
            # Extract date number from start_time format (DAYX_XXXXXXXX)
            if isinstance(start_time, str):
                start_time = int(start_time[3])  # Extract X from DAYX
            else:
                start_time = [int(t[3]) for t in start_time]  # Extract X from each DAYX
            if isinstance(start_time, list):
                import threading
                responses = []
                threads = []
                
                def query_thread(d):
                    response = self.query_day(d, keywords)
                    responses.append(response)
                
                for d in start_time:
                    thread = threading.Thread(target=query_thread, args=(d,))
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
                    
                return "\n".join(responses)
            else:
                return self.query_day(start_time, keywords)
        elif level == "hour":
            return self.query_hour(start_time, keywords)

    

    def query_videomme(self, level, keywords, query_time=None, start_time=None):
        second_data=[
            entry for entry in self.min_log
        ]
        min_data=[
            entry for entry in self.hour_log
        ]
        # Check and process start_time format
        if start_time is None:
            return "error: start_time is required"
            
        # Extract time component and validate HHMMSSFF format
        if '_' in start_time:
            # Format contains prefix and time component
            time_part = start_time.split('_')[-1]
            if not re.match(r'^\d{8}$', time_part):
                return "error: Invalid time format. Expected HHMMSSFF in time component"
        else:
            # Format is just HHMMSSFF
            if not re.match(r'^\d{8}$', start_time):
                return "error: Invalid time format. Expected HHMMSSFF"
            start_time = f"DAY1_{start_time}"  # Default to DAY1 if no prefix
        begin_time=start_time.split('_')[1]
        video_end_time=second_data[-1]['end_time']
        begin_time=scale_time_proportional(begin_time,video_end_time)
        filter_second_data=[
            entry for entry in self.min_log if int(entry['start_time'])>=int(begin_time) and int(entry['end_time'])<=(int(begin_time)+int("00100000"))]
        filter_min_data=min_data
        if len(filter_min_data)==0:
            return "No available data for Week level Query"
        if len(filter_second_data)==0:
            return "No available data for Hour level Query"
        if level=="week":
            prompt=f"10 minutes summaries {filter_min_data} \n keywords {keywords}"
            response_raw = call_gpt4(system_message=level1_prompt_for_videomme_long, prompt=prompt, max_tokens=800)
            response = json.loads(clean_json(strip_code_fences(response_raw)))
            try:
                first=response[0]
                first['level']='week'       
                first['exact_match']=True
            except:
                first=min_data[-1]
                if 'generated_text' in first:
                    first['relevant_content'] = first.pop('generated_text')
                first['exact_match']=False
                first['level']=level
            return first
        else:
            prompt=f"30 second summaries {filter_second_data} \n keywords {keywords}"
            response_raw = call_gpt4(system_message=level2_prompt_for_videomme_long, prompt=prompt, max_tokens=800)
            response = clean_json(strip_code_fences(response_raw))

            if response == "error":
                return "error: Search failed"
            else:
                response = json.loads(response)
            
            try:
                first=response[0]
                first['level']=level       
                first['exact_match']=True
            except:
                first=min_data[-1]
                if 'generated_text' in first:
                    first['relevant_content'] = first.pop('generated_text')
                first['exact_match']=False
                first['level']=level
            return first
    
    def query_egoschema(self, level, keywords, query_time=None, start_time=None):
        min_data=[
            entry for entry in self.min_log
        ]
        if len(min_data)==0:
            return "No available data for Hour level Query"
        prompt=f"30 Seconds summaries {min_data} \n keywords {keywords}"
        response_raw = call_gpt4(system_message=prompt_for_egoschema, prompt=prompt, max_tokens=800)
        response = strip_code_fences(response_raw)
        response = json.loads(strip_code_fences(response_raw))
        try:
            first=response[0]
            first['level']='hour'       
            first['exact_match']=True
        except:
            first=min_data[-1]
            if 'generated_text' in first:
                first['relevant_content'] = first.pop('generated_text')
            first['exact_match']=False
            first['level']='hour'
        return first
    
def scale_time_proportional(input_time_hhmmssff: str, target_duration_hhmmssff: str) -> str:
    """
    Scale a time point within a 24-hour period proportionally to a specified target duration.

    Args:
        input_time_hhmmssff (str): Source time point in 24-hour format (e.g., "12000000").
        target_duration_hhmmssff (str): Target total duration to map to (e.g., "00030000" for 3 minutes).

    Returns:
        str: The corresponding time point within the target duration.
    """
    fps = 30  # 固定帧率

    def to_frames(timecode: str) -> int:
        """Convert hhmmssff timecode to total frames."""
        hh = int(timecode[:2])
        mm = int(timecode[2:4])
        ss = int(timecode[4:6])
        ff = int(timecode[6:8])
        return ((hh * 3600 + mm * 60 + ss) * fps) + ff

    def to_timecode(frames: int) -> str:
        """Convert total frames to hhmmssff timecode."""
        total_seconds = frames // fps
        ff = frames % fps
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        return f"{hh:02d}{mm:02d}{ss:02d}{ff:02d}"

    # 1. Define total frames in 24-hour range
    frames_in_24_hours = 24 * 3600 * fps

    # 2. Calculate total frames in target duration
    frames_in_target_duration = to_frames(target_duration_hhmmssff)
    
    # If target duration is 0, return 0
    if frames_in_target_duration == 0:
        return "00000000"

    # 3. Convert source time point to frames
    input_frames = to_frames(input_time_hhmmssff)

    # 4. Calculate proportion of source time point in 24-hour range
    proportion = input_frames / frames_in_24_hours

    # 5. Apply proportion to target duration and round
    mapped_frames = round(proportion * frames_in_target_duration)

    # 6. Robustness check: ensure result doesn't exceed target total duration
    if mapped_frames >= frames_in_target_duration:
        mapped_frames = frames_in_target_duration - 1

    # 7. Convert final frames back to timecode and return
    return to_timecode(mapped_frames)

def _unescape_custom_json_sequences(match):
    """
    辅助函数，用于 re.sub，处理特定的转义序列。
    """
    sequence = match.group(0)
    if sequence == r"\'":  # Match literal \'
        return "'"
    elif sequence.startswith(r"\x"): # Match literal \xHH
        hex_code = sequence[2:]  # Extract HH part
        try:
            return chr(int(hex_code, 16)) # Convert to corresponding character
        except ValueError:
            # If HH is not a valid hexadecimal (theoretically guaranteed by regex), return original sequence
            return sequence
    return sequence # Should not reach here theoretically

def clean_json(raw_json_str: str) -> str:
    """
    """
    processed_string = raw_json_str

    try:
        # Step 1: Special handling for non-standard JSON escapes like \xHH and \'
        # Use regex r"\\x[0-9a-fA-F]{2}|\\'" to match literal \xHH or \' sequences.
        # This ensures only these specific sequences are replaced, not standard JSON escapes like \"
        processed_string = re.sub(
            r"\\x[0-9a-fA-F]{2}|\\'", # Note: backslash is literal, matches single backslash in input string
            _unescape_custom_json_sequences,
            processed_string
        )

        # Step 2: Replace U+00A0 (non-breaking space) with regular space.
        # This includes those generated from \xa0 (literal) -> U+00A0 (character) conversion,
        # or those originally existing as U+00A0 characters.
        # In Python string literals, U+00A0 character can be written as '\xa0'.
        processed_string = processed_string.replace('\xa0', ' ')

        # (Optional) Internal test: try parsing the processed string to verify its validity
        try:
            json.loads(processed_string)
            # If parsing succeeds, it means cleaning is effective
        except json.JSONDecodeError as e:
            # If parsing fails, print warning information
            print(f"Warning (internal test): Cleaned string still cannot be parsed by json.loads: {e}")
            print(f"Context at error position {e.pos}:")
            start = max(0, e.pos - 40)
            end = min(len(processed_string), e.pos + 40)
            print(f"...'{processed_string[start:e.pos]}<--Error occurred at-->{processed_string[e.pos:end]}'...")
            
        return processed_string

    except Exception as e:
        # Catch other errors that may occur in re.sub or other steps
        print(f"Error occurred while cleaning string: {e}")
        print(f"Returning original input string. Original string fragment (first 200 chars): '{raw_json_str[:200]}...'")
        return raw_json_str # As fallback, return original string
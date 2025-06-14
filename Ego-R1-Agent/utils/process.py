import re
import json
import requests
import ast
import logging
import utils.constants as constants

def parse_tool_call(query):
    # name_pattern = r'\'name\': ?\'?(.*?)\'?' 
    # r'\'name\': ?\'?(.*?)\']?'
    
    # if the output is a dict, try to load directly
    if isinstance(query, dict):
        query_info = query
        try:
            query_info['arguments'] = ast.literal_eval(query_info['arguments'])
            return query_info
        except:
            return None
        
    elif isinstance(query, str):
        try:
            query_info = ast.literal_eval(query)
            if 'name' in query_info or 'Name' in query_info or 'NAME' in query_info:
                if 'arguments' in query_info or 'Arguments' in query_info or 'ARGUMENTS' in query_info:
                    try:
                        query_info['arguments'] = ast.literal_eval(query_info['arguments'])
                        return query_info
                    except:
                        return {'name': query_info['name'], 'arguments': {query_info['arguments']}}
                else:
                    return {'name': query_info['name'], 'arguments': {[]}}
            else:
                return None
        except:
            # try to parse from string
            name_pattern = r"""['"]name['"]\s*:\s*['"]([^\'\"]+)['"]"""
            args_pattern = r"""['"]arguments['"]\s*:\s*({.*?})"""
            try:
                name = re.search(name_pattern, query, re.DOTALL).group(1)
                m = re.search(args_pattern, query, re.DOTALL)
                if not m:
                    raise ValueError("Couldn't find an arguments field")
                
                raw = m.group(1)
                
                arguments = json.loads(raw)
                
                return {'name': name, 'arguments': arguments}
            except:
                return None

def batch_video_llm(tool_queries, identity_list) -> list:
    payloads = []
    for tool, identity in zip(tool_queries, identity_list):
        if tool is None:
            payloads.append(None)
            continue
        if tool['name'] == 'video_llm':
            try:
                payloads.append({"question": tool['arguments']['question'], "range": tool['arguments']['range'], "identity": identity.split('_')[0]})
            except:
                payloads.append('')
        else:
            payloads.append(None)
    print("-"*20, payloads, "-"*20)
    results = requests.post(constants.VIDEO_LLM_URL, json=payloads, timeout=420).json()
    print("-"*20, results, "-"*20)
    
    logging.info(f"results: {results}")
    # extract the text answer
    for i, result in enumerate(results):
        if isinstance(result, dict):
            results[i] = str(result)
        else:
            continue
    return results

def tool_call(query_info, identity):
    # query_info = parse_tool_call(query)
    if query_info is None:
        return ''
    
    if 'rag' in query_info['name'].lower():
        if 'level' not in query_info['arguments'] or 'keywords' not in query_info['arguments'] or 'start_time' not in query_info['arguments'] or 'query_time' not in query_info['arguments']:
            return 'Invalid query arguments.'
        payload = {
            "level": query_info['arguments']['level'],
            "keywords": query_info['arguments']['keywords'],
            "start_time": query_info['arguments']['start_time'],
            "query_time": query_info['arguments']['query_time'],
        }
        # import ipdb; ipdb.set_trace()
        if identity not in ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]:
            payload['vid_id'] = query_info['vid_id']
        
        try:
            # import ipdb; ipdb.set_trace()
            results = requests.post(constants.RAG_URL[identity], json=payload).json()['response'] # ['relevant_content'], response has the day info and content
            results = str(results)
            # import ipdb; ipdb.set_trace()
            # results = str(requests.post(constants.RAG_URL[identity], json=payload))
        except Exception as e:
            results = ''
            print('Tool call error:', e, "Output:", results)
            
    elif 'video_llm' in query_info['name'].lower():
        if 'question' not in query_info['arguments'] or 'range' not in query_info['arguments']:
            results = 'Invalid query arguments.'
        else:
            if identity not in ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]:
                payload = {
                    "question": query_info['arguments']['question'],
                    "range": query_info['arguments']['range'],
                    "vid_id": query_info['vid_id']
                }
            else:
                payload = {
                    "question": query_info['arguments']['question'],
                    "range": query_info['arguments']['range'],
                    "identity": identity.split('_')[0]
                }
            # import ipdb; ipdb.set_trace()
            try:
                results = requests.post(constants.VIDEO_LLM_URL, json=[payload]).json()
            except:
                results = "API error, fail to query video_llm."
            
            if isinstance(results, list):
                if 'answer' in results[0]:
                    results = results[0]['answer']
                else:
                    results = results[0]
            else:
                results = "API error, fail to parse the response from video_llm."
            
    elif 'vlm' in query_info['name'].lower():
        if 'question' not in query_info['arguments'] or 'timestamp' not in query_info['arguments']:
            return 'Invalid query arguments.'
        if identity not in ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]:
            payload = {
                "question": query_info['arguments']['question'],
                "timestamp": query_info['arguments']['timestamp'],
                "vid_id": query_info['vid_id']
            }
        else:
            payload = {
                "question": query_info['arguments']['question'],
                "timestamp": query_info['arguments']['timestamp'],
                "identity": identity.split('_')[0]
            }
        try:
            results = requests.post(constants.VLM_URL, json=payload).json()
            if 'answer' in results:
                results = results['answer']
        except:
            results = 'API error.'
    else:
        print(f"Invalid tool name: {query_info['name']}.")
        return f"Invalid tool name: {query_info['name']}."
    
    return results


def parse_answer(text):
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    else:
        return None
    
def compute_score(pred, gt):
    score = 0.
    answer = parse_answer(pred)

    if answer is None:
        return score
    
    pattern = r"\s*([A-Z])\s*"
    match = re.search(pattern, answer, re.DOTALL)
    try:
        answer = match.group(1)
        if answer.strip().lower() == gt.strip().lower():
            score = 1.
    except:
        pass

    return score


def get_query(text):
    import re
    pattern = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None
    
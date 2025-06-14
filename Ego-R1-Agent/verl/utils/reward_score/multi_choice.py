import re
import json
import wandb

def parse_answer(text):
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    else:
        return None

def compute_format_score(solution_str, format_score_pos=0.5, format_score_neg=0.):
    special_tokens = ['<think>', '</think>', '<answer>', '</answer>', '<tool>', '</tool>']

    def found_special_token(s):
        for token in special_tokens:
            if token in s:
                return True
        return False
    
    def parse_tool_call(query):
        # name_pattern = r'\'name\': ?\'?(.*?)\'?' 
        # r'\'name\': ?\'?(.*?)\']?'
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

    solution_str = solution_str.strip()
    while solution_str.strip():
        if solution_str.startswith('<think>'):
            solution_str = solution_str[len('<think>'):]
        else:
            return format_score_neg
        
        end_think_pos = solution_str.find('</think>')
        if end_think_pos == -1:
            return format_score_neg
        
        if found_special_token(solution_str[:end_think_pos]): # should not contain any special tokens inside each block
            return format_score_neg
        
        solution_str = solution_str[end_think_pos + len('</think>'):].strip()
        if solution_str.startswith('<answer>'):
            solution_str = solution_str[len('<answer>'):]
            end_answer_pos = solution_str.find('</answer>')
            if end_answer_pos == -1:
                return format_score_neg
            
            if found_special_token(solution_str[:end_answer_pos]):
                return format_score_neg

            solution_str = solution_str[end_answer_pos + len('</answer>'):].strip()
            if found_special_token(solution_str): # stop after answer block
                return format_score_neg
            else:
                return format_score_pos
            
        elif solution_str.startswith('<tool>'):
            solution_str = solution_str[len('<tool>'):]
            end_tool_pos = solution_str.find('</tool>')
            if end_tool_pos == -1:
                return format_score_neg
            
            if found_special_token(solution_str[:end_tool_pos]):
                return format_score_neg

            tool_call = parse_tool_call(solution_str[:end_tool_pos])
            if tool_call is None:
                return format_score_neg
            
            solution_str = solution_str[end_tool_pos + len('</tool>'):].strip()
            if solution_str.startswith('<information>'):
                end_information_pos = solution_str.find('</information>')
                if end_information_pos == -1:
                    return format_score_neg
                
                if found_special_token(solution_str[:end_information_pos]):
                    return format_score_neg

                solution_str = solution_str[end_information_pos + len('</information>'):].strip()
                

        else:
            return format_score_neg
    return format_score_pos

def compute_score(solution_str, ground_truth, format_score=0.5):
    score = 0.
    answer = parse_answer(solution_str)

    if answer is None:
        return score
    
    pattern = r"\s*([A-D])\s*"
    match = re.search(pattern, answer, re.DOTALL)
    
    try:
        answer = match.group(1)
        if answer.strip().lower() == ground_truth['target'].strip().lower():
            score = 1.
    except:
        pass

    if score == 0.:
        score = compute_format_score(solution_str, format_score_pos=format_score, format_score_neg=0.)
        
        # logging to wandb
        # wandb.log({
        #     "format_score": score,
        # })

    return score

multi_choice_test_str = """
<think>think about it</think>
<tool>
{
    "name": "tool_name",
    "arguments": {
        "arg1": "arg1_value",
        "arg2": "arg2_value",
        "arg3": [1, 2]
    }
}
</tool>
<information>information</information>
<think>think about it</think>
<answer>A</answer>
"""

if __name__ == "__main__":
    # solution_str = "<think>think about it</think> <tool>use the tool</tool> <information>information</information> <think>think about it</think> <tool>use the tool</tool> <information>information</information> <think>think about it</think> <answer>A</answer>."
    print(compute_score(multi_choice_test_str, {'target': 'B'}))
    print(compute_score(multi_choice_test_str, {'target': 'A'}))


    # solution_str = "I think the answer is <answer>A</answer>."
    # print(compute_format_score(solution_str))

import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import json

import utils.process as process
import utils.constants as constants

USE_BATCH_VIDEO_LLM = True
@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

USE_NEW_SFT = True

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</tool>')[0] + '</tool>'
                 if '</tool>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences self.tokenizer.decode(original_right_side['responses'][0], skip_special_tokens=False)
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]} # self.tokenizer.decode(original_left_side['input_ids'][0], skip_special_tokens=False)
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        
        # add identity to the rollings
        identity_list = gen_batch.non_tensor_batch['identity']
        
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        
        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            # import ipdb; ipdb.set_trace() # rollings.batch = self.tensor_fn.cut_to_effective_len(rollings.batch,keys=['input_ids', 'attention_mask', 'position_ids'])
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # import ipdb; ipdb.set_trace()
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings) self.tokenizer.decode(next_obs_ids[0], skip_special_tokens=True)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info             # self.tokenizer.decode(responses_ids[0], skip_special_tokens=False) # (responses_ids[0] == self.tokenizer.pad_token_id).nonzero()[0].item()
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses']) # <|im_end|>\n
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # NEW SFT
            if USE_NEW_SFT:
                new_suffix_pad = torch.tensor([self.tokenizer.pad_token_id] * 2).unsqueeze(0).repeat(len(responses_ids), 1)
                responses_ids = torch.cat([responses_ids, new_suffix_pad], dim=1)
                for i in range(len(responses_ids)):
                    # try:
                    pad_start = (responses_ids[i] == self.tokenizer.pad_token_id).nonzero()[0].item()
                    responses_ids[i][pad_start:pad_start+2] = torch.tensor(self.tokenizer('<|im_end|>\n')['input_ids'])
                # except:
                #     import ipdb; ipdb.set_trace()
                # import ipdb; ipdb.set_trace()
                # responses_ids[0][245:247] = torch.tensor(self.tokenizer('<|im_end|>\n')['input_ids'])
                # responses_ids[0][225:227] = torch.tensor(self.tokenizer('<|im_end|>\n')['input_ids'])
            # NEW SFT
            # import ipdb; ipdb.set_trace()
            print("RESPONSES_STR:", responses_str)
            
            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, identity_list
            )
            
            print("NEXT_OBS:", next_obs)
            print("DONES:", dones)
            print("VALID_ACTION:", valid_action)
            print("IS_SEARCH:", is_search)
            
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            # <|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n
            next_obs_ids = self._process_next_obs(next_obs) # self.tokenizer.decode(next_obs_ids[0], skip_special_tokens=False)
            # NEW SFT
            if USE_NEW_SFT:
                if step == self.config.max_turns - 1:
                    new_suffix_pad = torch.tensor([self.tokenizer.pad_token_id] * 2).unsqueeze(0).repeat(len(next_obs_ids), 1)
                else:
                    new_suffix_pad = torch.tensor([self.tokenizer.pad_token_id] * 5).unsqueeze(0).repeat(len(next_obs_ids), 1)
                user_prefix = torch.tensor(self.tokenizer('<|im_start|>user\n')['input_ids'])
                user_prefix = user_prefix.repeat(len(next_obs_ids), 1)
                next_obs_ids = torch.cat([user_prefix, next_obs_ids, new_suffix_pad], dim=1)
                for i in range(len(next_obs_ids)):
                    pad_start = (next_obs_ids[i] == self.tokenizer.pad_token_id).nonzero()[0].item()
                    if step == self.config.max_turns - 1:
                        next_obs_ids[i][pad_start:pad_start+2] = torch.tensor(self.tokenizer('<|im_end|>\n')['input_ids'])
                    else:
                        next_obs_ids[i][pad_start:pad_start+5] = torch.tensor(self.tokenizer('<|im_end|>\n<|im_start|>assistant\n')['input_ids'])
            # NEW SFT

            # import ipdb; ipdb.set_trace()
            
            # Update states (rollings.batch['attention_mask'][0] == 0).nonzero().tolist()
            rollings = self._update_rolling_state( # self.tokenizer.decode(rollings.batch['input_ids'][0], skip_special_tokens=False)
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side( # self.tokenizer.decode(original_right_side['responses_with_info_mask'][0], skip_special_tokens=False)
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            # NEW SFT
            # import ipdb; ipdb.set_trace()
            # DO NOT ADD STOP PROMPT IN THE LAST STEP
            # if USE_NEW_SFT: # self.tokenizer.decode(resp_prefix[0], skip_special_tokens=False)
            #     resp_prefix = self.tokenizer('<|im_start|>assistant\n' + constants.stop_prompt, add_special_tokens=False, return_tensors='pt')['input_ids'].repeat(len(active_mask), 1)
            #     rollings = self._update_rolling_state( # self.tokenizer.decode(rollings.batch['input_ids'][0], skip_special_tokens=False)
            #         rollings,
            #         resp_prefix,
            #         torch.tensor([], dtype=torch.int64).repeat(len(active_mask), 1)
            #     )
            # self.tokenizer.decode(r2.batch['input_ids'][0], skip_special_tokens=False)
            # NEW SFT r2 = self._update_rolling_state(rollings, resp_prefix, torch.tensor([], dtype=torch.int64).repeat(len(active_mask), 1))

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            print("RESPONSES_STR:", responses_str)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, identity_list, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            # NEW SFT
            # if USE_NEW_SFT: 
            #     responses_ids = torch.concat([resp_prefix, responses_ids], dim=1)
            # NEW SFT
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, identity_list=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []

        tool_queries = [process.parse_tool_call(content) for action, content in zip(cur_actions, contents) if action == 'tool']
        # tool_queries = [process.parse_tool_call(content) if action == 'tool' else None for action, content in zip(cur_actions, contents)]
        
        if do_search:
            tool_results = self.batch_tool_call(tool_queries, identity_list)
            try:
                assert len(tool_results) == sum([1 for action in cur_actions if action == 'tool']), print(len(tool_results), sum([1 for action in cur_actions if action == 'tool']))
            except:
                import ipdb; ipdb.set_trace()
        else:
            tool_results = [''] * sum([1 for action in cur_actions if action == 'tool'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'tool':
                    try:
                        next_obs.append(f'<information>{tool_results.pop(0).strip()}</information>')
                        # next_obs.append(f'\n\n<information>{tool_results[0].strip()}</information>\n\n')
                        # tool_results.pop(0)
                    except:
                        # import ipdb; ipdb.set_trace()
                        next_obs.append(f'<information>My previous tool call didn\'t return any information. Let me try again.</information>')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to call a tool, I should put the tool call with name and arguments between <tool> and </tool>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(tool_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(tool|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]
    
    def batch_tool_call(self, tool_queries, identity_list):
        # import ipdb; ipdb.set_trace()
        results = []
        if USE_BATCH_VIDEO_LLM:
            results = process.batch_video_llm(tool_queries, identity_list) # len(results) == batch_size
            assert len(results) == len(tool_queries), print(results, tool_queries)
            for i, (item, identity) in enumerate(zip(tool_queries, identity_list)):
                if item is None or item == '':
                    results[i] = ''
                    continue
                if isinstance(item, dict) and item['name'] == 'video_llm':
                    continue
                tool_result = process.tool_call(item, identity)         
                results[i] = tool_result
        else:
            for item, identity in zip(tool_queries, identity_list):
                if item is None:
                    results.append('')
                    continue
                tool_result = process.tool_call(item, identity)
                results.append(tool_result)
                
        # for result in results:
        #     if result is None:
        #         result = ''
        #     if not "<information>" in result and not "</information>" in result:
        #         result = f"<information>{result}</information>"
        return results
        
    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

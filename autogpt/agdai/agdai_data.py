''' Data and communication module for AGDAI plugin'''
import os
import time
import re
import difflib
import random
import json
from enum import Enum
# from typing import List #, Any, Dict, Optional, TypedDict, TypeVar, Tuple
from autogpt.singleton import AbstractSingleton
from autogpt.config import Config
from autogpt.llm.llm_utils import create_chat_completion
from .agdai_mem import ClAgdaiMem, ClAgdaiVals
from .telegram_chat import TelegramUtils

class ClAgdaiData(AbstractSingleton):
    def __init__(self) -> None:
        super().__init__()
        self._curr_agent_name = ''
        self._utc_start = int(time.time())
        self._seqnum = 0
        self._contexts = ClAgdaiMem(self._utc_start, 'contexts')
        self._actions = ClAgdaiMem(self._utc_start, 'actions')
        self._advice = ClAgdaiMem(self._utc_start, 'advice')
        self._response_refs = ClAgdaiVals(self._utc_start, 'respose_refs', val_type='tuple') #Note. Not using refs but rather the generic vals
        self._action_scores = ClAgdaiVals(self._utc_start, 'action_scores')
        self._advice_refs = ClAgdaiVals(self._utc_start, 'advice_refs', val_type='tuple')
        self._advice_scores = ClAgdaiVals(self._utc_start, 'advice_scores', val_type='float')
        self._advice_source = ClAgdaiVals(self._utc_start, 'advice_source', val_type='str')
        self._telegram_api_key = os.getenv("TELEGRAM_API_KEY")
        self._telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self._telegram_utils = TelegramUtils(
                self._telegram_api_key, self._telegram_chat_id,
                [   ('score', 'send agent a score -10 -> 10 to express satisfaction'),
                    ('score_1', 'send agent a score -10 -> 10 to express your opinion about last action only'),
                    ('advice', 'send agent advice as to how to proceed'),
                    ('request_logs', 'ask agent for the last <n> logs, defaults to 100')])
        self.init_bot()
        self.msg_user('System initialized')

    def set_agent_name(self, agent_name : str):
        self._curr_agent_name = agent_name
        # self.add_mem(f'new agent: {agent_name}', [], 0)

    def init_bot(self):
        self._telegram_utils.init_commands()
        self._telegram_utils.ignore_old_updates()

    class UserCmd(Enum):
        eScore = 1
        eAdvice = 2
        eGetLogs = 3
        eFreeForm = 4
        eScore_1 = 5

    def parse_user_msg(self, user_str):
        pattern = r"^/score_?1?\s+(-?\d+)"
        match = re.search(pattern, user_str, re.IGNORECASE)
        if match:
            try:
                score = int(match.group(1))
                if "/score_1" in user_str.lower():
                    cmd = self.UserCmd.eScore_1
                else:
                    cmd = self.UserCmd.eScore
                return cmd, max(-10, min(score, 10))
            except ValueError:
                return ''
        pattern = r"^/advice\s+(.+)"
        match = re.search(pattern, user_str, re.IGNORECASE)
        if match:
            return self.UserCmd.eAdvice, match.group(1)
        pattern = r"^/logs\s*(\d*)$"
        match = re.search(pattern, user_str, re.IGNORECASE)
        if match:
            if match.group(1):
                return self.UserCmd.eGetLogs, int(match.group(1))
            else:
                return self.UserCmd.eGetLogs, 100
        return self.UserCmd.eFreeForm, user_str

    c_mem_tightness = 0.1 # TBD Make configurable

    def create_helpful_input(self, context_as_str, context_embedding) -> str:
        top_k = int(self._contexts.get_numrecs()) // 10 # TBD Make configurable
        top_memids = self._contexts.get_topk(context_embedding, top_k)
        if random.random() < 0.5:
            suggestion = self.get_previous_advice(top_memids)
            if suggestion is not None and len(suggestion) > 0:
                return suggestion
        
        return self.get_good_memory(context_as_str, top_memids)

    def get_context_diff_summary(self, context_as_str, top_memid) -> str:
        top_context = self._contexts.get_text(top_memid)
        context_diff = difflib.unified_diff(context_as_str.splitlines(), top_context.splitlines())
        now_str, top_str = '', ''
        for line in context_diff:
            if line[0] == '-' and line[1] != '-':
                now_str += line[1:] + '\n'
            elif line[0] == '+' and line[1] != '+':
                top_str += line[1:] + '\n'

        if len(now_str) < 1 and len(top_str) < 1:
            return ''
        
        sys_prompt = 'Your task is summarize in a few sentences the key differences '\
                + 'between the current context and a previous context\n'
        user_prompt = f'current context:\n{now_str}'\
                f'previous context:\n{top_str}'
        cfg = Config()
        messages = [
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
        diff_response = create_chat_completion(messages, cfg.fast_llm_model)

        return '. However these are the differences between the current context '\
            f'and the one where this action was successful\n: {diff_response}'

    def find_closest_advice_ref(self, amemid, distance):
        '''
        Find an associated advice, iterating both forward and backwards
            in time withing the same start utc set of records
        '''
        bck_memids = self._contexts.get_bck_memids(amemid, distance)
        fwd_memids = self._contexts.get_fwd_memids(amemid, distance)
        for i in range(distance):
            if i < len(bck_memids):
                context_memid = bck_memids[i]
                val_ret = self._advice_refs.get_val(context_memid)
                if val_ret is not None:
                    return val_ret
            if i < len(fwd_memids):
                context_memid = fwd_memids[i]
                val_ret = self._advice_refs.get_val(context_memid)
                if val_ret is not None:
                    return val_ret
        return None

        
    def get_previous_advice(self, top_memids) -> str:
        advice_memids = []; advice_scores = []
        for amemid in top_memids:
            val_ret = self.find_closest_advice_ref(amemid, 5) # TBD: Make magic number configurable
            if val_ret is None:
                continue
            advice_utc, advice_seq = val_ret
            # assert((context_utc, context_seq) == amemid)
            advice_memids.append((advice_utc, advice_seq))
            rando = (random.random() - 0.5) * self.c_mem_tightness
            advice_scores.append(self._advice_scores.get_val((advice_utc, advice_seq)) + rando)
        if len(advice_scores) < 1:
            return ''
        imax = advice_scores.index(max(advice_scores))
        best_advice = self._advice.get_text(advice_memids[imax])
        # if advice_scores[imax] < 2:
        #     return ''
        reminder = '''The following advice was given to me by my user in a different context.\n
I should try and follow my user\'s advice but realize that it may have been given in a different context.\n
I must make sure I use the json format specified above for my response.
'''
        return f'Here is an example of advice that my user sent to me in the past: \n{best_advice}\n{reminder}'

    def get_good_memory(self, context_as_str, top_memids):
        '''
        Current thinking:
        Get 1/10th of the most relevant context records
        Find the action with the highest score, perhaps weighted by closeness ranking
        Sort the closest actions to it and apply their ranking 
        Take a diff between the closest context and the current context
        Call the model to list the differences
        present the best action
        '''
        action_memids = []; action_scores = []; context_memids = []
        for amemid in top_memids:
            val_ret = self._response_refs.get_val(amemid)
            if val_ret is None:
                continue
            action_utc, action_seq = val_ret
            # assert((context_utc, context_seq) == amemid)
            action_memids.append((action_utc, action_seq))
            rando = (random.random() - 0.5) * self.c_mem_tightness
            action_scores.append(self._action_scores.get_val((action_utc, action_seq)) + rando)
            context_memids.append(amemid)
        # if len(action_scores) < 1:
        #     return ''
        imax = action_scores.index(max(action_scores))
        best_action = self._actions.get_text(action_memids[imax])
        if action_scores[imax] < 4:
            return ''
        '''
        The way to improve this is to create a ranking for each of action of how close each other
        action is. Then add the other scores weighted by reciprocal ranking within the set of 
        close contexts
        '''
        
        '''
        First stage, I'm just going to put the best action into the message stream.
        Actually, I need to compare the winning context with the current context and create an
        explanation which precedes the winning action for how this differs from the current context,
        Current thoughts on this is to throw out all lines from the context that are identical 
        using difflib.
        Build a prompt out of these differences and ask GPT to list the differences. Then add that into 
        the preabmble before presenting a past action.
        '''
        diff_summary = self.get_context_diff_summary(context_as_str, context_memids[imax])
        if diff_summary is None or len(diff_summary) < 1:
            diff_summary = 'in a different context.'

        reminder = f'This past response was successful {diff_summary}. \n'\
                'I should consider making a similar response but adapt it to the task at hand. \n'\
                'I must make sure I use the json format specified above for my response.'
        return f'Here is an example of successful response that I made in the past: \n{best_action}\n{reminder}'

    def apply_scores(self, start_back: int = 0, num_back: int = 10, score : float = 0):
        """
        Apply scores to a memory table (such as actions or advice) going back in time
        Scores are only applied to the current run
        Args:
            start_back. If zero,  starting at the top of the table, else start this number back
            num_back. Apply with decreasing factor as many entries back as num_back.
                        Note num_back is counted from the last entry, even if start_back is > 0
            score. Value to add to current value (score)
        """
        score_frac = score
        for memid in self._actions.get_inseq_memids(num_back, start_back):
            rec_score = self._action_scores.get_val(memid)
            # The policy being tried out here is to assume that any score 
            # goes back only as far as some other score but cannot replace
            # it or add to it.
            if rec_score != 0:
                break
            self._action_scores.set_val(memid, rec_score + score_frac)
            score_frac *= 0.8 # TBD make user settable, configurable or something
            del memid

        for memid in self._contexts.get_inseq_memids(num_back, start_back):
            retval = self._advice_refs.get_val(memid)
            if retval is None:
                continue
            utc_advice, seqnum_advice = retval
            self._advice_scores.set_val((utc_advice, seqnum_advice), score)
            break

        

    def process_actions(self, gpt_response : str) -> str:
        '''
        Currently does nothing other than store the GPT response
        Future versions may compare to guidelines or apply some other processing.
        Return value is the response that we want app to run with
        '''
        gpt_response_json = json.dumps(gpt_response) # TBD is the argument a string?
        last_action_memid, _ = self._actions.add(gpt_response_json)
        last_context_memid = self._contexts.get_last_memid()
        self._response_refs.add((last_context_memid, last_action_memid))
        self._action_scores.add((last_action_memid, 0))
        return gpt_response
    
    def store_advice(self, advice : str, source : str) -> None:
        '''
        Stores the advice created by your user, internal processing or a third party
        Stores a pointer from the context to the advice as well as from the advice
            to the score and the advice source
        '''
        advice_memid, _ = self._advice.add(advice)
        last_context_memid = self._contexts.get_last_memid()
        self._advice_refs.add((last_context_memid, advice_memid))
        self._advice_scores.add((advice_memid, 0))
        self._advice_source.add((advice_memid, source))

    def process_msgs(self, messages : dict[str, str]):
        # new_messages = [msg for msg in messages if msg not in self._full_message_history]
        # look for "Error:"" in last msg
        if len(messages) > 2 and 'Error:' in messages[-3]['content']:
            self.apply_scores(0, 1, -7)
        context_as_str = '\n'.join([f'{key} {value}' for message_dict in messages \
                                    for key, value in message_dict.items()])
        _, context_embedding = self._contexts.add(context_as_str)
        user_message = self._telegram_utils.check_for_user_input()
        if len(user_message) > 0:
            msg_type, content = self.parse_user_msg(user_message)
            if msg_type == self.UserCmd.eGetLogs:
                self.msg_user('logs..logs..')
                return ''
            elif msg_type == self.UserCmd.eScore:
                self.apply_scores(0, 10, content)
                # return ''
            elif msg_type == self.UserCmd.eScore_1:
                self.apply_scores(0, 1, content)
                # return ''
            elif msg_type == self.UserCmd.eAdvice:
                self.store_advice(content, 'user')
                return f'Your user has requested that you use the following advice in deciding on your future responses: {content}'
            elif msg_type == self.UserCmd.eFreeForm:
                return f'Your user has sent you the following message: {content}\n\
Use the command "telegram_message_user" from the COMMANDS list if you wish to reply.\n\
Ensure your response uses the JSON format specified above.'

        if self._contexts.get_numrecs() > 10:
            return self.create_helpful_input(context_as_str, context_embedding)

        return ''



    def msg_user(self, message):
        return self._telegram_utils.send_message(message)

    def check_for_user_message(self):
        return self._telegram_utils.check_for_user_input()

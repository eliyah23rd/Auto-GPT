''' Data and communication module for AGDAI plugin'''
import os
import time
import re
import difflib
import random
import json
from enum import Enum
from typing import List, Tuple #, Any, Dict, Optional, TypedDict, TypeVar
from autogpt.singleton import AbstractSingleton
from .agdai_mem import ClAgdaiMem, ClAgdaiRefs, ClAgdaiVals
from .telegram_chat import TelegramUtils

class ClAgdaiData(AbstractSingleton):
    def __init__(self) -> None:
        super().__init__()
        self._curr_agent_name = ''
        self._utc_start = int(time.time())
        self._seqnum = 0
        self._contexts = ClAgdaiMem(self._utc_start, 'contexts')
        self._actions = ClAgdaiMem(self._utc_start, 'actions')
        self._response_refs = ClAgdaiVals(self._utc_start, 'respose_refs', val_type='tuple') #Note. Not using refs but rather the generic vals
        self._action_scores = ClAgdaiVals(self._utc_start, 'action_scores')
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

    def get_good_memory(self, context_embedding):
        '''
        Current thinking:
        Get 1/10th of the most relevant context records
        Find the action with the highest score, perhaps weighted by closeness ranking
        Sort the closest actions to it and apply their ranking 
        Take a diff between the closest context and the current context
        Call the model to list the differences
        present the best action
        '''
        c_mem_tighness = 0.1 # TBD Make configurable
        top_k = int(self._contexts.get_numrecs()) // 10 # TBD Make configurable
        top_memids = self._contexts.get_topk(context_embedding, top_k)
        action_memids = []; action_scores = []
        for amemid in top_memids:
            (action_utc, action_seq) = self._response_refs.get_val(amemid)
            # assert((context_utc, context_seq) == amemid)
            action_memids.append((action_utc, action_seq))
            rando = (random.random() - 0.5) * c_mem_tighness
            action_scores.append(self._action_scores.get_val((action_utc, action_seq)) + rando)
        imax = action_scores.index(max(action_scores))
        best_action = self._actions.get_text(action_memids[imax])
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
        # d = difflib.unified_diff(text1, text2, lineterm='')
        reminder = '''This past action was successful in a different context. 
Pay attention to the task you are trying to achieve and make sure you use the json format specified above for your response.
'''
        return f'Here is an example of successful response that you made in the past: \n{best_action}\n{reminder}'

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
            self._action_scores.set_val(memid, rec_score + score_frac)
            score_frac *= 0.8 # TBD make user settable, configurable or something

    def process_actions(self, gpt_response : str) -> str:
        '''
        Currently does nothing other than store the GPT response
        Future versions may compare to guidelines or apply some other processing.
        Return value is the response that we want app to run with
        '''
        gpt_response_json = json.dumps(gpt_response)
        last_action_memid, _ = self._actions.add(gpt_response_json)
        last_context_memid = self._contexts.get_last_memid()
        self._response_refs.add((last_context_memid, last_action_memid))
        self._action_scores.add((last_action_memid, 0))
        return gpt_response

    def process_msgs(self, messages : dict[str, str]):
        # new_messages = [msg for msg in messages if msg not in self._full_message_history]
        # look for "Error:"" in last msg
        if len(messages) > 2 and 'Error:' in messages[-3]['content']:
            self.apply_scores(0, 1, -7)
        _, context_embedding = self._contexts.add(\
                '\n'.join([f'{key} {value}' for message_dict in messages for key, value in message_dict.items()])) 
        user_message = self._telegram_utils.check_for_user_input()
        if len(user_message) > 0:
            msg_type, content = self.parse_user_msg(user_message)
            if msg_type == self.UserCmd.eGetLogs:
                self.msg_user('logs..logs..')
                return ''
            elif msg_type == self.UserCmd.eScore:
                self.apply_scores(0, 10, content)
                return ''
            elif msg_type == self.UserCmd.eScore_1:
                self.apply_scores(0, 1, content) 
                return ''
            elif msg_type == self.UserCmd.eAdvice:
                return f'Your user has requested that you use the following advice in deciding on your future responses: {content}'
            elif msg_type == self.UserCmd.eFreeForm:
                return f'Your user has sent you the following message: {content}\n\
Use the command "telegram_message_user" from the COMMANDS list if you wish to reply.\n\
Ensure your response uses the JSON format specified above.'

        if self._contexts.get_numrecs() > 10:
            return self.get_good_memory(context_embedding)

        return ''



    def msg_user(self, message):
        return self._telegram_utils.send_message(message)

    def check_for_user_message(self):
        return self._telegram_utils.check_for_user_input()

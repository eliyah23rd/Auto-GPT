''' Data and communication module for AGDAI plugin'''
import os
import time
import re
import difflib
import random
import json
from enum import Enum
from colorama import Fore # , Style
# from typing import List #, Any, Dict, Optional, TypedDict, TypeVar, Tuple
from autogpt.singleton import AbstractSingleton
from autogpt.config import Config, AIConfig
from autogpt.logs import logger
from autogpt.ai_guidelines import AIGuidelines
from autogpt.llm.utils import create_chat_completion
from autogpt.llm.base import ChatSequence, Message
from .agdai_mem import ClPAIMem, ClPAIVals
from .telegram_chat import TelegramUtils
class ClPAIData(AbstractSingleton):
    def __init__(self, config : Config = None, ai_config = None) -> None:
        super().__init__()
        assert config is not None, 'The first call must pass the config'
        self._config : Config = config # Config()
        self._ai_config : AIConfig = ai_config
        cfg = self._config
        self._curr_agent_name = ''
        self._utc_start = int(time.time())
        self._guidelines_mgr = AIGuidelines(cfg.ai_guidelines_file)
        self._seqnum = 0
        self._contexts = ClPAIMem(self._utc_start, 'contexts')
        self._actions = ClPAIMem(self._utc_start, 'actions')
        self._advice = ClPAIMem(self._utc_start, 'advice')
        self._response_refs = ClPAIVals(self._utc_start, 'respose_refs', val_type='tuple') #Note. Not using refs but rather the generic vals
        self._action_scores = ClPAIVals(self._utc_start, 'action_scores')
        self._advice_refs = ClPAIVals(self._utc_start, 'advice_refs', val_type='tuple')
        self._advice_scores = ClPAIVals(self._utc_start, 'advice_scores', val_type='float')
        self._advice_source = ClPAIVals(self._utc_start, 'advice_source', val_type='str')
        self._helpful_hints = ClPAIVals(self._utc_start, 'helpful_hints', val_type='str')
        self._telegram_api_key = os.getenv("TELEGRAM_API_KEY")
        self._telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self._telegram_utils = TelegramUtils(
                self._telegram_api_key, self._telegram_chat_id,
                [   ('score', 'send agent a score -10 -> 10 to express satisfaction'),
                    ('score_1', 'send agent a score -10 -> 10 to express your opinion about last action only'),
                    ('advice', 'send agent advice as to how to proceed'),
                    ('history', 'ask agent for data <n> cycles back. Prints context, action, price and score'),
                    ('pause', 'tells agent to pause before the next GPT iteration. Will not continue until given advice or unpaused'),
                    ('unpause', 'unpaused agent after pause, resumes the interaction with underlying GPT engine'),
                    ('intervene', 'after an action is selected by GPT check if user wants to override. Maintains this state till flow command'),
                    ('flow', 'if in intervene state, returns to GPT selected action'),
                    ('ask', 'asks the agent a question'),
                    ])
        self._bintervene = False # When switched to True by the /intervene command, allows user to replace GPT response
        self._b_last_fixed = True # If there is not an uninterrupted sequence of fixed, new fixes are not allowed
        self._b_action_response = False # must be set to True before any calls to modules that are making mainstream calls to the LLM and reset afterwards
        self.init_bot()
        self.msg_user('System initialized')

    def set_agent_name(self, agent_name : str):
        self._curr_agent_name = agent_name
        # self.add_mem(f'new agent: {agent_name}', [], 0)

    def init_bot(self):
        self._telegram_utils.init_commands()
        self._telegram_utils.ignore_old_updates()

    def register_cmds(self, cmds):
        for cmd in cmds:
            self._ai_config.command_registry.register(cmd)

    class UserCmd(Enum):
        eScore                  = 1
        eAdvice                 = 2
        eGetHistory             = 3
        eFreeForm               = 4
        eScore_1                = 5
        ePause                  = 6
        eContinueAfterPause     = 7
        eIntervene              = 8
        eFlow                   = 9

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
        pattern = r"^/history\s*(\d*)$"
        match = re.search(pattern, user_str, re.IGNORECASE)
        if match:
            if match.group(1):
                return self.UserCmd.eGetHistory, int(match.group(1))
            else:
                return self.UserCmd.eGetHistory, 0
        pattern = r"/pause\s*$"
        match = re.search(pattern, user_str, re.IGNORECASE)
        if match:
            return self.UserCmd.ePause, ''
        pattern = r"/unpause\s*$"
        match = re.search(pattern, user_str, re.IGNORECASE)
        if match:
            return self.UserCmd.eContinueAfterPause, ''
        pattern = r"/intervene\s*$"
        match = re.search(pattern, user_str, re.IGNORECASE)
        if match:
            return self.UserCmd.eIntervene, ''
        pattern = r"/flow\s*$"
        match = re.search(pattern, user_str, re.IGNORECASE)
        if match:
            return self.UserCmd.eFlow, ''
        return self.UserCmd.eFreeForm, user_str

    c_mem_tightness = 0.1 # TBD Make configurable

    def create_helpful_input(self, context_as_str, context_embedding) -> str:
        numrecs = int(self._contexts.get_numrecs())
        assert numrecs > 1, 'This function may only be called after checking the size of the db'
        top_k = min(10, numrecs - 1) # int(self._contexts.get_numrecs()) // 10 # TBD Make configurable
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
            if len(line) < 3:
                continue
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
        # cfg = Config()
        diff_messages = ChatSequence.for_model(
            self._config.fast_llm_model,
            [
                Message(
                    "system",
                    sys_prompt,
                ),
                Message(
                    "user",
                    user_prompt,
                ),
            ]
        )
        diff_response = create_chat_completion(
            prompt=diff_messages,
            config=self._config,
            model=self._config.fast_llm_model,
        )
        
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

    def store_best_action(self, best_action, diff_summary):
        context_memid = self._contexts.get_last_memid()
        rec = f'Best action: {best_action}\n Diff summary: {diff_summary}'
        self._helpful_hints.add((context_memid, rec))

    def get_previous_advice(self, top_memids) -> str:
        advice_memids = []; advice_scores = []
        # The following mitigates the effect of the reciprocal later so that instead of 1/2:1, 1/7:1/6
        c_recip_mitigator = 5 # TBD make configurable
        for idist, amemid in enumerate(top_memids):
            val_ret = self.find_closest_advice_ref(amemid, 5) # TBD: Make magic number configurable
            if val_ret is None:
                continue
            advice_utc, advice_seq = val_ret
            # assert((context_utc, context_seq) == amemid)
            advice_memids.append((advice_utc, advice_seq))
            rando = (random.random() - 0.5) * self.c_mem_tightness
            advice_score = self._advice_scores.get_val((advice_utc, advice_seq))
            advice_score *= 1 / (c_recip_mitigator + idist)
            advice_scores.append(advice_score + rando)
        if len(advice_scores) < 1:
            return ''
        imax = advice_scores.index(max(advice_scores))
        best_advice = self._advice.get_text(advice_memids[imax])
        if advice_scores[imax] < 1:
            return ''
        self.store_advice(best_advice, '_retrieved')
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
        # The following mitigates the effect of the reciprocal below so that instead of 1/2:1, 1/7:1/6
        c_recip_mitigator = 5 # TBD make configurable
        action_memids = []; action_scores = []; context_memids = []
        for idist, amemid in enumerate(top_memids):
            val_ret = self._response_refs.get_val(amemid)
            if val_ret is None:
                continue
            action_utc, action_seq = val_ret
            # assert((context_utc, context_seq) == amemid)
            action_memids.append((action_utc, action_seq))
            rando = (random.random() - 0.5) * self.c_mem_tightness
            action_score = self._action_scores.get_val((action_utc, action_seq))
            action_score = 0 if action_score is None else action_score
            action_score *= 1 / (c_recip_mitigator + idist)
            action_scores.append(action_score + rando)
            context_memids.append(amemid)
        if len(action_scores) < 1:
            return ''
        imax = action_scores.index(max(action_scores))
        best_action = self._actions.get_text(action_memids[imax])
        if action_scores[imax] < 1:
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

        self.store_best_action(best_action, diff_summary)

        reminder = f'This past response was successful {diff_summary}. \n'\
                'I should consider making a similar response but adapt it to the task at hand. \n'\
                'I must make sure I use the json format specified above for my response.'
        return f'Here is an example of successful response that I made in the past: \n{best_action}\n{reminder}'

    def apply_scores(self, start_back: int = 0, num_back: int = 10, score : float = 0, b_force_score = False):
        """
        Apply scores to a memory table (such as actions or advice) going back in time
        Scores are only applied to the current run
        Args:
            start_back. If zero,  starting at the top of the table, else start this number back
            num_back. Apply with decreasing factor as many entries back as num_back.
                        Note num_back is counted from the last entry, even if start_back is > 0
            score. Value to add to current value (score)
            b_force_score. If True, writes a score even if there was already a score assigned
        """
        score_frac = score
        for memid in self._actions.get_inseq_memids(num_back, start_back):
            rec_score = self._action_scores.get_val(memid)
            # The policy being tried out here is to assume that any score 
            # goes back only as far as some other score but cannot replace
            # it or add to it.
            if rec_score is not None and not b_force_score:
                continue
            self._action_scores.set_val(memid, score_frac) # rec_score + 
            score_frac *= 0.8 # TBD make user settable, configurable or something
            del memid

        for memid in self._contexts.get_inseq_memids(num_back, start_back):
            retval = self._advice_refs.get_val(memid)
            if retval is None:
                continue
            utc_advice, seqnum_advice = retval
            self._advice_scores.set_val((utc_advice, seqnum_advice), score)
            break

        

    def process_actions(self, gpt_response_json : str) -> str:
        '''
        If we are not in the intervene state do nothing other than store the GPT response
        Future versions may compare with guidelines or apply some other processing.
        If we are in the intervene state we start a conversation with the user
            that will let them replace the GPT answer with their own
        Return value is the response that we want app to run with
        '''
        # gpt_response_json : str = json.dumps(gpt_response)

        # before anything else, make sure that this is called by an on_respose from a 
        #   mainstream LLM call (not a guidelines or summary call)
        if not self._b_action_response:
            return gpt_response_json
        
        self._b_action_response = False
        baccept = True
        if self._bintervene:
            baccept = False
            self.msg_user('LLM made the following response:')
            self.msg_user(gpt_response_json)
            self.msg_user('Please type \'y\' if you want to accept this or \'n\' to change it')
            while True:
                y_or_n_str = self._telegram_utils.check_for_user_input(bblock=True)
                if y_or_n_str.lower().strip() == 'y':
                    baccept = True
                    break
                elif y_or_n_str.lower().strip() == 'n':
                    break
                else:
                    self.msg_user('I\'m sorry, please type \'y\' if you want to accept this or \'n\' to change it')
            if not baccept:
                json_str = '{"thoughts": {"text": "", "reasoning": "", "plan": "", "criticism": "", "speak": ""}, ' \
                        + '"command": {"name": "' # cmd_str", "args": {"filename": "sample.txt"}}}'
                self.msg_user('Please enter the command name.')
                cmd_str = self._telegram_utils.check_for_user_input(bblock=True)
                json_str += cmd_str.strip() + '", "args": {"'
                b_first_arg = True
                while True:
                    self.msg_user('Please enter the name of the next argument or type \'end\' if there are no more arguments.')
                    arg_name = self._telegram_utils.check_for_user_input(bblock=True)
                    if len(arg_name) < 1 or arg_name.lower().strip() == 'end':
                        break
                    if b_first_arg:
                        b_first_arg = False
                    else:
                        json_str += ', '
                    arg_name = arg_name.strip()
                    json_str += arg_name + '": "'
                    self.msg_user(f'Please enter the value for arg \'{arg_name}\'.')
                    arg_val = self._telegram_utils.check_for_user_input(bblock=True)
                    json_str += arg_val.strip() + '"'
                json_str += '}}}'
                gpt_response_json = json_str.replace('\n', '\\n')
                # gpt_response = json.loads(gpt_response_json)
                baccept = True

        if baccept:
            last_action_memid, _ = self._actions.add(gpt_response_json, self._config)
            last_context_memid = self._contexts.get_last_memid()
            self._response_refs.add((last_context_memid, last_action_memid))
            self._action_scores.add((last_action_memid, None)) # NB None and not zero for action score
            self._b_last_fixed = False
            return gpt_response_json
        
        assert False, 'logic should not reach here'
    
    def store_advice(self, advice : str, source : str) -> None:
        '''
        Stores the advice created by your user, internal processing or a third party
        Stores a pointer from the context to the advice as well as from the advice
            to the score and the advice source
        '''
        advice_memid, _ = self._advice.add(advice)
        last_context_memid = self._contexts.get_last_memid()
        self._advice_refs.add((last_context_memid, advice_memid))
        self._advice_scores.add((advice_memid, 0)) # NB 0 and not None for advice score
        self._advice_source.add((advice_memid, source))

    def get_history(self, num_back):
        msg = ''
        context_memid = self._contexts.get_inseq_memids(1 ,num_back)[0]
        context_text = self._contexts.get_text(context_memid)
        if context_text is not None:
            msg += f'Context:\n{context_text}\n'
        action_memid = self._response_refs.get_val(context_memid)
        action_score = None
        if action_memid is not None:
            action_text = self._actions.get_text(action_memid)
            if action_text is not None:
                msg += f'LLM response:\n{action_text}\n'
            action_score = self._action_scores.get_val(action_memid)
        advice_memid = self._advice_refs.get_val(context_memid)
        if advice_memid is not None:
            advice_text = self._advice.get_text(advice_memid)
            if advice_memid is not None:
                msg += f'advice applied:\n{advice_text}\n'
        hint_text = self._helpful_hints.get_val(context_memid)
        if hint_text is not None:
            msg += f'Helpful hint added: \n{hint_text}\n'
        if action_score is not None:
            msg += f'score for action:\n{action_score}'

        self.msg_user(msg)

    def process_msgs(self, messages : list[dict[str, str]]):
        # new_messages = [msg for msg in messages if msg not in self._full_message_history]
        # look for "Error:"" in last msg
        bviolation, violation_alert = self._guidelines_mgr.exec_monitor(self._config, messages)
        if bviolation:
            self.apply_scores(0, 10, -10, b_force_score=True)
            # In the future we will investigate more carefully
            self.msg_user(f'Guideline violation alert! Shutting down! Report: {violation_alert}')
            # raise ValueError('Guideline violation!') TBD!!! Don't forget to bring this back

        if len(messages) > 2 and 'Error:' in messages[-3]['content']:
            self.apply_scores(0, 1, -7)
        context_as_str = '\n'.join([f'{key} {value}' for message_dict in messages \
                                    for key, value in message_dict.items()])
        _, context_embedding = self._contexts.add(context_as_str, self._config)
        bpaused = False
        while True:
            user_message = self._telegram_utils.check_for_user_input()
            if len(user_message) > 0:
                msg_type, content = self.parse_user_msg(user_message)
                if msg_type == self.UserCmd.eGetHistory:
                    self.get_history(content)
                elif msg_type == self.UserCmd.eScore:
                    self.apply_scores(0, 10, content)
                elif msg_type == self.UserCmd.eScore_1:
                    self.apply_scores(0, 1, content)
                elif msg_type == self.UserCmd.eIntervene:
                    self._bintervene = True
                elif msg_type == self.UserCmd.eFlow:
                    self._bintervene = False
                elif msg_type == self.UserCmd.ePause:
                    bpaused = True
                elif msg_type == self.UserCmd.eContinueAfterPause:
                    bpaused = False
                elif msg_type == self.UserCmd.eAdvice:
                    self.store_advice(content, 'user')
                    return f'Your user has requested that you use the following advice in deciding on your future responses: {content}'
                elif msg_type == self.UserCmd.eFreeForm:
                    return f'Your user has sent you the following message: {content}\n'\
                        'Use the command "telegram_message_user" from the COMMANDS list if you wish to reply.\n'\
                        'Ensure your response uses the JSON format specified above.'
            if not bpaused:
                break
            time.sleep(2)
                
        ret_text = ''
        if self._contexts.get_numrecs() > 3:
            hint_text = self.create_helpful_input(context_as_str, context_embedding)
            if len(hint_text) > 3:
                # logger.typewriter_log('HINT:', Fore.YELLOW, f"\n\n\n{hint_text}")
                print(f'HINT:\n\n\n{hint_text}')
                ret_text = hint_text

        self._b_action_response = True
        return ret_text



    def msg_user(self, message):
        return self._telegram_utils.send_message(message)

    def check_for_user_message(self):
        return self._telegram_utils.check_for_user_input()

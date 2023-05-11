''' Data and communication module for AGDAI plugin'''
import os
import time
import json
from typing import List, Tuple #, Any, Dict, Optional, TypedDict, TypeVar
from autogpt.singleton import AbstractSingleton
from .agdai_mem import ClAgdaiMem
from .telegram_chat import TelegramUtils

class ClAgdaiData(AbstractSingleton):
    def __init__(self) -> None:
        super().__init__()
        self._curr_agent_name = ''
        self._utc_start = int(time.time())
        self._seqnum = 0
        self._history = ClAgdaiMem(self._utc_start)
        self._full_message_history = []
        self._full_history_memids = []
        self._telegram_api_key = os.getenv("TELEGRAM_API_KEY")
        self._telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self._telegram_utils = TelegramUtils(
                self._telegram_api_key, self._telegram_chat_id)
        self.msg_user('System initialized')

    def add_mem(self, text: str, refs: List[Tuple[int, int]], score):
        self._history.add(text, self._curr_agent_name, (self._utc_start, self._seqnum), refs, score)
        self._seqnum += 1

    def set_agent_name(self, agent_name : str):
        self._curr_agent_name = agent_name
        self.add_mem(f'new agent: {agent_name}', [], 0)

    def add_msg(self, amem, all_refs, msg_refs, score):
        aref = (self._utc_start, self._seqnum)
        self.add_mem(amem, all_refs + msg_refs, score)
        msg_refs.append(aref)

    def process_msgs(self, messages : List[str]):
        all_refs = []
        # new_messages = [msg for msg in messages if msg not in self._full_message_history]
        new_messages = []
        for msg in messages:
            if msg in self._full_message_history:
                all_refs.extend(self._full_history_memids[self._full_message_history.index(msg)])
            else:
                new_messages.append(msg)
        for event in new_messages:
            self._full_message_history.append(event)
            msg_refs = []
            if event["role"].lower() == "assistant":
                # Remove "thoughts" dictionary from "content"
                content_dict = json.loads(event["content"])
                for key, val in content_dict.items():
                    if isinstance(val, str):
                        self.add_msg(f'Your {key}: {val}', all_refs, msg_refs, 0)
                    elif isinstance(val, dict):
                        for key2, val2 in val.items():
                            if isinstance(val2, str):
                                self.add_msg(f'Your {key2} in {key}: {val2}', all_refs, msg_refs, 0)

                # event["content"] = json.dumps(content_dict)

            elif event["role"].lower() == "system":
                content = event["content"]
                self.add_msg(f"your computer said: {content}", all_refs, msg_refs, 0)

            self._full_history_memids.append(msg_refs)
            all_refs += msg_refs

            # Delete all user messages
            # elif event["role"] == "user":
            #     new_events.remove(event)
        user_message = self._telegram_utils.check_for_user_input()
        if len(user_message) > 0:
            return f'Your user has sent you the following message: {user_message}\n\
Use the command "telegram_message_user" from the COMMANDS list if you wish to reply.\n\
Ensure your response uses the JSON format specified above.'
        else:
            return ''


    def msg_user(self, message):
        return self._telegram_utils.send_message(message)

    def check_for_user_message(self):
        return self._telegram_utils.check_for_user_input()

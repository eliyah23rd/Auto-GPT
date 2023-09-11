""" Autonamous Guidelines Driven AI Command Functions"""
# from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar

import os
from autogpt.agent.agent import Agent
from .dospai_data import ClDOSPAIData

def _is_file_binary_fn(file_path: str):
    """Given a file path load all its content and checks if the null bytes is present

    Args:
        file_path (_type_): _description_

    Returns:
        bool: is_binary
    """
    with open(file_path, "rb") as f:
        file_data = f.read()
    if b"\x00" in file_data:
        return True
    return False

def _dospai_read_file(filename: str, agent: Agent) -> str:
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"read_file {filename} failed: no such file or directory"
        )
    if not _is_file_binary_fn(filename):
        with open(filename, 'rt', encoding="utf-8") as fh:
            return fh.read()


def _dospai_find_similar(memory : str, agent: Agent) -> str:
    '''
    Returns information quoting and scoring various memories
    '''
    pai_data = ClDOSPAIData(config=agent.config) # , ai_config=agent.ai_config)
    print(f'_dospai_find_similar called to find memories similar to {memory}')

def _dospai_msg_user(message : str, agent: Agent) -> None:
    pai_data = ClDOSPAIData(config=agent.config) # , ai_config=agent.ai_config)
    return pai_data.msg_user(message)

def _dospai_ask_gpt(prompt : str, memslot : str, agent: Agent) -> None:
    pai_data = ClDOSPAIData(config=agent.config) # , ai_config=agent.ai_config)
    pai_data._b_gpt_function = True
    return pai_data.ask_gpt(prompt, memslot)

def _dospai_store_memslot(memslot : str, agent: Agent) -> None:
    pai_data = ClDOSPAIData(config=agent.config) # , ai_config=agent.ai_config)
    return pai_data.store_memslot(memslot)


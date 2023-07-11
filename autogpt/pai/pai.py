""" Autonamous Guidelines Driven AI Command Functions"""
# from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar

from autogpt.agent.agent import Agent
from .pai_data import ClPAIData

def _pai_find_similar(memory : str, agent: Agent) -> str:
    '''
    Returns information quoting and scoring various memories
    '''
    pai_data = ClPAIData(config=agent.config, ai_config=agent.ai_config)
    print(f'_pai_find_similar called to find memories similar to {memory}')

def _pai_msg_user(message : str, agent: Agent) -> None:
    pai_data = ClPAIData(config=agent.config, ai_config=agent.ai_config)
    return pai_data.msg_user(message)

def _pai_ask_gpt(prompt : str, memslot : str, agent: Agent) -> None:
    pai_data = ClPAIData(config=agent.config, ai_config=agent.ai_config)
    return pai_data.ask_gpt(prompt, memslot)

def _pai_store_memslot(memslot : str, agent: Agent) -> None:
    pai_data = ClPAIData(config=agent.config, ai_config=agent.ai_config)
    return pai_data.store_memslot(memslot)


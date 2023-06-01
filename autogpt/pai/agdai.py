""" Autonamous Guidelines Driven AI Command Functions"""
# from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar

from .agdai_data import ClPAIData

def _pai_find_similar(memory : str) -> str:
    '''
    Returns information quoting and scoring various memories
    '''
    pai_data = ClPAIData()
    print(f'_pai_find_similar called to find memories similar to {memory}')

def _pai_msg_user(message : str) -> None:
    pai_data = ClPAIData()
    return pai_data.msg_user(message)

""" Autonamous Guidelines Driven AI Command Functions"""
# from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar

from .agdai_data import ClAgdaiData

def _agdai_find_similar(memory : str) -> str:
    '''
    Returns information quoting and scoring various memories
    '''
    agdai_data = ClAgdaiData()
    print(f'agdai find similar called to find memories similar to {memory}')

def _agdai_msg_user(message : str) -> None:
    agdai_data = ClAgdaiData()
    return agdai_data.msg_user(message)

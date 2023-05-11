from __future__ import annotations

import os
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar

import numpy as np
import orjson

from autogpt.llm import get_ada_embedding
from autogpt.memory.base import MemoryProviderSingleton

EMBED_DIM = 1536
SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS


def create_default_embeddings():
    return np.zeros((0, EMBED_DIM)).astype(np.float32)


@dataclasses.dataclass
class AgdaiCacheContent:
    texts: List[str] = dataclasses.field(default_factory=list)
    agent_names: List[str] = dataclasses.field(default_factory=list)
    memids: List[Tuple[int, int]] = dataclasses.field(default_factory=list)
    refs: List[List[Tuple[int, int]]] = dataclasses.field(default_factory=list)
    rev_refs: List[List[Tuple[int, int]]] = dataclasses.field(default_factory=list)
    scores: List[float] = dataclasses.field(default_factory=list)
    seq_starts: Dict[str, int] = dataclasses.field(default_factory=dict)
    embeddings: np.ndarray = dataclasses.field(
        default_factory=create_default_embeddings
    )


class ClAgdaiMem:
    """A class that stores the memory in a local file"""

    def __init__(self, utc_start) -> None:
        """Initialize a class instance

        Args:
            cfg: Config object

        Returns:
            None
        """
        curr_dir = Path(__file__).parent
        self.filename = curr_dir / 'memory.json'

        self.filename.touch(exist_ok=True)

        if os.path.exists(self.filename):
            try:
                # Modified this to load permanent memory
                # - this can be cleared later
                with open(self.filename, "a+b") as f:
                    f.seek(0)
                    file_content = f.read()
                    if not file_content.strip():
                        file_content = b"{}"
                        f.write(file_content)

                    loaded = orjson.loads(file_content)
                    # embeddings need to be converted back to numpy array
                    # to align with later use.
                    if 'embeddings' in loaded:
                        loaded['embeddings'] = np.array(loaded['embeddings'])
                    self.data = AgdaiCacheContent(**loaded)
            except orjson.JSONDecodeError:
                print(f"Error: The file '{self.filename}' is not in JSON format.")
                self.data = AgdaiCacheContent()
        else:
            file_content = b"{}"
            with self.filename.open("w+b") as f:
                f.write(file_content)

            self.data = AgdaiCacheContent()

        self.data.seq_starts[str(utc_start)] = len(self.data.texts)

    def add(self, text: str, agent_name: str, memid: Tuple[int, int], refs: List[Tuple[int, int]], score):
        """
        Add text to our list of texts, add embedding as row to our
            embeddings-matrix

        Args:
            text: str

        Returns: None
        """
        if "Command Error:" in text:
            return ""
        self.data.texts.append(text)
        self.data.agent_names.append(agent_name)

        embedding = get_ada_embedding(text)

        vector = np.array(embedding).astype(np.float32)
        vector = vector[np.newaxis, :]
        self.data.embeddings = np.concatenate(
            [
                self.data.embeddings,
                vector,
            ],
            axis=0,
        )
        self.data.memids.append(memid)
        self.data.refs.append(refs)
        self.data.rev_refs.append([])
        self.data.scores.append(score)
        for utc_start, seqnum in refs:
            idx = self.data.seq_starts[str(utc_start)] + seqnum
            self.data.rev_refs[idx].append(memid)
            


        with open(self.filename, "wb") as f:
            out = orjson.dumps(self.data, option=SAVE_OPTIONS)
            f.write(out)
        return text

    def clear(self) -> str:
        """
        Clears the data in memory.

        Returns: A message indicating that the memory has been cleared.
        """
        self.data = AgdaiCacheContent()
        return "Obliviated"

    def get(self, data: str) -> list[Any] | None:
        """
        Gets the data from the memory that is most relevant to the given data.

        Args:
            data: The data to compare to.

        Returns: The most relevant data.
        """
        return self.get_relevant(data, 1)

    def get_relevant(self, text: str, k: int) -> list[Any]:
        """ "
        matrix-vector mult to find score-for-each-row-of-matrix
         get indices for top-k winning scores
         return texts for those indices
        Args:
            text: str
            k: int

        Returns: List[str]
        """
        embedding = get_ada_embedding(text)

        scores = np.dot(self.data.embeddings, embedding)

        top_k_indices = np.argsort(scores)[-k:][::-1]

        return [self.data.texts[i] for i in top_k_indices]

    def get_stats(self) -> tuple[int, tuple[int, ...]]:
        """
        Returns: The stats of the local cache.
        """
        return len(self.data.texts), self.data.embeddings.shape

'''
            # Remove "thoughts" dictionary from "content"
            content_dict = json.loads(event["content"])
            if "thoughts" in content_dict:
                del content_dict["thoughts"]
            event["content"] = json.dumps(content_dict)
'''
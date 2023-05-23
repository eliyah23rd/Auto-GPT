from __future__ import annotations

import os
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Tuple #  TypedDict, TypeVar, Optional

import numpy as np
import json
import orjson

from autogpt.llm import get_ada_embedding
# from autogpt.memory.base import MemoryProviderSingleton

EMBED_DIM = 1536
SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS # pylint: disable no-member


def create_default_embeddings():
    return np.zeros((0, EMBED_DIM)).astype(np.float32)


@dataclasses.dataclass
class AgdaiCacheContent:
    texts: List[str] = dataclasses.field(default_factory=list)
    embeddings: np.ndarray = dataclasses.field(
        default_factory=create_default_embeddings
    )
    seq_starts: Dict[str, int] = dataclasses.field(default_factory=dict)
    main_data_name: str = 'texts'


@dataclasses.dataclass
class AgdaiCacheRefs:
    refs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = dataclasses.field(default_factory=list) # list of src->dest memids where each memid is (start, seqnum)
    seq_starts: Dict[str, int] = dataclasses.field(default_factory=dict)
    main_data_name: str = 'refs'


@dataclasses.dataclass
class AgdaiCacheVals:
    vals: List[Tuple[Tuple[int, int], Any]] = dataclasses.field(default_factory=list) # list of src -> int vals where each src is a memid (start, seqnum)
    seq_starts: Dict[str, int] = dataclasses.field(default_factory=dict)
    main_data_name: str = 'vals'

# def init_file(utc_start, memtype):

class ClAgdaiStorage:
    def __init__(self, utc_start, memtype, filename_body) -> None:
        self._utc_start = str(utc_start)
        curr_dir = Path(__file__).parent
        self.filename = curr_dir / f'{filename_body}.json'

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
                    self.data = memtype(**loaded)
            except orjson.JSONDecodeError:
                print(f"Error: The file '{self.filename}' is not in JSON format.")
                self.data = memtype()
                # self._numrecs = 0
        else:
            file_content = b"{}"
            with self.filename.open("w+b") as f:
                f.write(file_content)

            self.data = memtype()
            # self._numrecs = 0

        self.data.seq_starts[self._utc_start] = self.get_numrecs()
        # self._numrecs = len(self.data[self.data.main_data_name])

    # def _incr_numrecs(self):
    #     # self._numrecs += 1
    #     raise NotImplementedError('deprecated')

    def save_data(self):
        with open(self.filename, "wb") as f:
            out = orjson.dumps(self.data, option=SAVE_OPTIONS)
            f.write(out)

    def get_last_memid(self):
        '''
        Get the last memid in the current utc
        '''
        return (self._utc_start, (self.get_numrecs() - 1) - self.data.seq_starts[self._utc_start])

    def _get_last_seq_memid(self, amemid):
        '''
        Given a memid, find its utc and then get the last memid
        with that utc
        '''
        lstarts, d_rev = self._get_utc_prep()
        lstarts.sort()
        utc, _ = amemid
        this_start = self.data.seq_starts[utc]
        lstarts_idx = lstarts.index(this_start)
        if lstarts_idx == len(lstarts) - 1:
            return self.get_last_memid()
        utc_next = d_rev[lstarts[lstarts_idx+1]]
        return utc, self.data.seq_starts[utc_next] - self.data.seq_starts[utc] - 1

    def get_inseq_memids(self, n, start_back : int = 0):
        '''
        Get memids from the current utc
        Starts at startback from the the last (i.e. startback=0 means the last)
        Goes back and gets n from the end, ignoring start_back
        '''
        _, num_inseq = self.get_last_memid()
        src_seq = max(0, num_inseq - start_back)
        range_upper = src_seq + 1
        range_lower = min(max(0, (src_seq - n)+1), range_upper-1)

        return [(self._utc_start, i) for i in reversed(range(range_lower, range_upper))] # If the numbers were bigger, this should be a yield
    
    def get_bck_memids(self, src_memid, n):
        '''
        Given a memid, retreive it and n-1 previous memid withing the same seq
        Order is src_memid first
        '''
        utc, src_seq = src_memid
        range_lower = max(0, (src_seq - n)+1)
        range_upper = src_seq + 1
        return [(utc, i) for i in reversed(range(range_lower, range_upper))] # If the numbers were bigger, this should be a yield
    
    def get_fwd_memids(self, src_memid, n):
        '''
        Given a memid, retreive it and n-1 next memid withing the same seq
        Order is src_memid first
        '''
        _, num_inseq = self._get_last_seq_memid(src_memid)
        utc, src_seq = src_memid
        range_upper = min(num_inseq+1, src_seq + n)
        return [(utc, i) for i in range(src_seq, range_upper)] # If the numbers were bigger, this should be a yield
    
    def get_numrecs(self):
        raise NotImplementedError('Subclass must define the get_numrecs function')
    
    def _get_utc_prep(self):
        lstarts = []; d_rev = {}
        for utc, idx_start in self.data.seq_starts.items():
            lstarts.append(idx_start)
            d_rev[idx_start] = utc

        return lstarts, d_rev

    def _highest_number_lte(self, lst, num):
        return max([i for i in lst if i <= num])

    def get_memids(self, idxs):
        lstarts, d_rev = self._get_utc_prep()
        lret = []
        for idx in idxs:
            seq_start = self._highest_number_lte(lstarts, idx)
            lret.append((d_rev[seq_start], idx - seq_start))
        return lret


class ClAgdaiVals(ClAgdaiStorage):
    """A class that stores key,val pairs in a local json file"""

    def __init__(self, utc_start, filename_body, val_type=None) -> None:
        super().__init__(utc_start, AgdaiCacheVals, filename_body)
        self.data.vals = [(tuple(key), tuple(val) if val_type == 'tuple' else val) for key, val in self.data.vals]
        self._d_lkp = {val[0]:idx for idx, val in enumerate(self.data.vals)}
        assert len(self._d_lkp) == self.get_numrecs(), 'Error, ClAgdaiVals may not include repeated key'

    def add(self, mem_with_score: Tuple[Tuple[int, int], Any]):
        """
        Add a mapping of a memid to storage. e.g. memid for action to its utility function score

        Args:
            mem_with_score: the memid of a record and an attached score

        Returns: None
        """
        key = mem_with_score[0]
        assert self._d_lkp.get(key, None) is None, 'Error, ClAgdaiVals may not include repeated key'
        self._d_lkp[key] = self.get_numrecs()
        self.data.vals.append(mem_with_score)
        # self._incr_numrecs()
        self.save_data()

    def get_val(self, memid) -> Any:
        idx = self._d_lkp.get(memid, None)
        if idx is None:
            return None
        lkp_memid, val = self.data.vals[idx]
        assert lkp_memid == memid
        return val
    
    def set_val(self, memid, new_val) -> None:
        idx = self._d_lkp.get(memid, None)
        assert idx is not None, 'Error! memid to set val for, not found in storage'
        lkp_memid, _ = self.data.vals[idx]
        assert lkp_memid == memid
        self.data.vals[idx] = (memid, new_val)
        self.save_data()

    def get_numrecs(self):
        return len(self.data.vals)


class ClAgdaiRefs(ClAgdaiStorage):
    """A class that stores ref pairs in a local json file"""

    def __init__(self, utc_start, filename_body) -> None:
        super().__init__(utc_start, AgdaiCacheRefs, filename_body)

    def add(self, ref_pair: Tuple[Tuple[int, int], Tuple[int, int]]):
        """
        Add a pair of memids to storage

        Args:
            ref_pair: a pair of memids from two separate storage files

        Returns: None
        """
        self.data.refs.append(ref_pair)
        # self._incr_numrecs()
        self.save_data()

    def get_ref_pairs(self, n):
        return [self.data.get_pair_ref(memid) for memid in self.get_inseq_memids(n)]

    def get_pair_ref(self, memid) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        utc, seqnum = memid
        idx = self.data.seq_starts[utc] + seqnum
        return self.data.refs[idx]

    def get_numrecs(self):
        return len(self.data.refs)


class ClAgdaiMem(ClAgdaiStorage):
    """A class that stores text plus embeddings in a local json file"""

    def __init__(self, utc_start, filename_body) -> None:
        """Initialize a class instance

        Args:
            utc_start: utc time in seconds prior to init
            memtype: a dataclass class (not instance)

        Returns:
            None
        """
        super().__init__(utc_start, AgdaiCacheContent, filename_body)
        # self.data, self.filename = init_file(utc_start, memtype)


    def add(self, text: str):
        """
        Add text to our list of texts, add embedding as row to our
            embeddings-matrix

        Args:
            text: str

        Returns: None
        """
        self.data.texts.append(text)

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
        # self._incr_numrecs()
        self.save_data()
        # return (self._utc_start, len(self.data.texts) - self.data.seq_starts[self._utc_start])
        # return the memid just created
        return self.get_last_memid(), embedding

    def clear(self) -> str:
        """
        Clears the data in memory.

        Returns: A message indicating that the memory has been cleared.
        """
        self.data = AgdaiCacheContent()
        return "Obliviated"

    def get_text(self, memid) -> str:
        utc, seqnum = memid
        idx = self.data.seq_starts[utc] + seqnum
        return self.data.texts[idx]

    def get_recs(self, n):
        return [self.data.texts[i] for i in self.get_inseq_memids(n)]


    def get(self, data: str) -> list[Any] | None:
        """
        Gets the data from the memory that is most relevant to the given data.

        Args:
            data: The data to compare to.

        Returns: The most relevant data.
        """
        return self.get_relevant(data, 1)

    def _get_topk(self, embedding: np.ndarray,  k: int) -> list[Any]:
        scores = np.dot(self.data.embeddings, embedding)
        # The following is far more efficient for large arrays than using argsort
        if 
        top_idxs = np.argpartition(scores, -(k+1))[-(k+1):]
        top_scores = scores[top_idxs]
        top_top_idxs = np.argsort(-top_scores)
        top_idxs = top_idxs[top_top_idxs]
        return top_idxs[1:]



    def get_topk(self, embedding: np.ndarray,  k: int) -> list[Any]:
        idxs = self._get_topk(embedding, k)
        return self.get_memids(idxs)

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

        top_k_indices = self._get_topk(embedding, k)

        return [self.data.texts[i] for i in top_k_indices]

    def get_stats(self) -> tuple[int, tuple[int, ...]]:
        """
        Returns: The stats of the local cache.
        """
        return len(self.data.texts), self.data.embeddings.shape

    def get_numrecs(self):
        return len(self.data.texts)


'''
            # Remove "thoughts" dictionary from "content"
            content_dict = json.loads(event["content"])
            if "thoughts" in content_dict:
                del content_dict["thoughts"]
            event["content"] = json.dumps(content_dict)
'''


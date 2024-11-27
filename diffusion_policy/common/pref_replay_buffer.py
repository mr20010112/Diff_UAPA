from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property

def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0

def rechunk_recompress_array(group, name, 
        chunks=None, chunk_length=None,
        compressor=None, tmp_key='_temp'):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)
    
    if compressor is None:
        compressor = old_arr.compressor
    
    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr

def get_optimal_chunks(shape, dtype, 
        target_chunk_bytes=2e6, 
        max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape)-1
    for i in range(len(shape)-1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[:i+1])
        if this_chunk_bytes <= target_chunk_bytes \
            and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(this_max_chunk_length, math.ceil(
            target_chunk_bytes / item_chunk_bytes))
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks

class PrefReplayBuffer:
    """
    Zarr-based temporal data structure specifically for preference dataset.
    Stores pairs of trajectories (observations, actions) along with votes.
    """
    
    def __init__(self, root: Union[zarr.Group, Dict[str, dict]]):
        """
        Initialize the preference replay buffer. Use class methods to create or load buffers.
        """
        assert 'data' in root
        assert 'meta' in root
        for key, value in root['data'].items():
            assert value.shape[0] == root['meta']['votes'].shape[0]
        self.root = root

    # ============= Create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        data = root.require_group('data', overwrite=False)
        meta = root.require_group('meta', overwrite=False)
        
        if 'votes' not in meta:
            votes = meta.zeros('votes', shape=(0,), dtype=np.float32, compressor=None, overwrite=False)
        if 'votes_2' not in meta:
            votes_2 = meta.zeros('votes_2', shape=(0,), dtype=np.float32, compressor=None, overwrite=False)
        
        return cls(root=root)

    @classmethod
    def create_empty_numpy(cls):
        root = {
            'data': dict(),
            'meta': {
                'episode_ends': np.zeros((0,), dtype=np.int64),
                'votes': np.zeros((0,), dtype=np.float32), 
                'votes_2': np.zeros((0,), dtype=np.float32),
                'length': np.zeros((0,), dtype=np.int64),
                'length_2': np.zeros((0,), dtype=np.int64),
                'beta_priori': np.zeros((0,), dtype=np.float32),
                'beta_priori_2': np.zeros((0,), dtype=np.float32),
            }
        }
        return cls(root=root)

    @classmethod
    def create_from_group(cls, group, **kwargs):
        if 'data' not in group:
            # create from scratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exists
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode='r', **kwargs):
        """
        Open a Zarr file from disk for large datasets that cannot fit in memory.
        """
        group = zarr.open(zarr_path, mode=mode)
        return cls.create_from_group(group=group, **kwargs)

    # ============= Add episodes ===============
    def add_pref_episode(self, data: Dict[str, np.ndarray], 
                         meta_data: Optional[Dict[str, Union[np.ndarray, int]]] = None,
                         chunks: Optional[Dict[str, tuple]] = dict(),
                         compressors: Union[str, numcodecs.abc.Codec, dict] = dict()):
        """
        Add a pair of episodes (obs/action for each trajectory) along with metadata (votes).
        """
        assert 'obs' in data and 'obs_2' in data, "obs and obs_2 keys are required"
        assert 'action' in data and 'action_2' in data, "action and action_2 keys are required"
        
        is_zarr = isinstance(self.root, zarr.Group)
        curr_len = len(self.root['meta']['votes'])
        episode_length = len(data['obs'])
        new_len = curr_len + 1

        # Add trajectory 1
        for key in ['obs', 'action']:
            # Create the new shape to accommodate all time steps
            value = data[key]
            new_shape = (new_len,) + (episode_length,) + data[key].shape[1:]  # This will set (new_len, T, dim)

            if key not in self.root['data']:
                # Create a new array if it doesn't exist
                if is_zarr:
                    cks = self._resolve_array_chunks(chunks, key, data[key])
                    cpr = self._resolve_array_compressor(compressors, key, data[key])
                    arr = self.root['data'].zeros(name=key, shape=new_shape, chunks=cks, dtype=data[key].dtype, compressor=cpr)
                else:
                    arr = np.zeros(new_shape, dtype=data[key].dtype)
                    self.root['data'][key] = arr
            else:
                arr = self.root['data'][key]
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)

            # Store the full sequence, adjusting the shape to match the time steps in data[key]
            arr[new_len-1, -value.shape[0]:, :] = value # Now this assumes data[key] has shape (T, dim)


        # Add trajectory 2 (obs_2, action_2)
        for key in ['obs_2', 'action_2']:
            value = data[key]
            # Create the new shape to accommodate all time steps
            new_shape = (new_len,) + (episode_length,) + data[key].shape[1:]  # This will set (new_len, T, dim)

            if key not in self.root['data']:
                # Create a new array if it doesn't exist
                if is_zarr:
                    cks = self._resolve_array_chunks(chunks, key, data[key])
                    cpr = self._resolve_array_compressor(compressors, key, data[key])
                    arr = self.root['data'].zeros(name = key, shape=new_shape, chunks=cks, dtype=data[key].dtype, compressor=cpr)
                else:
                    arr = np.zeros(new_shape, dtype=data[key].dtype)
                    self.root['data'][key] = arr
            else:
                arr = self.root['data'][key]
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)

            # Store the full sequence, adjusting the shape to match the time steps in data[key]
            arr[new_len-1, -value.shape[0]:, :] = value  # Now this assumes data[key] has shape (T, dim)

        # Add votes to meta
        if meta_data:
            for key in ['votes', 'votes_2']:
                new_shape = (new_len,) + (1,)
                if key not in self.root['meta']:
                    if is_zarr:
                        self.root['meta'].zeros(name=key, shape=new_shape, chunks=new_shape, dtype=np.float32)
                    else:
                        self.root['meta'][key] = np.zeros(new_shape, dtype=np.float32)
                arr = self.root['meta'][key]
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
                arr[new_len-1] = meta_data[key]

            for key in ['length', 'length_2']:
                new_shape = (new_len,) + (1,)
                if key not in self.root['meta']:
                    if is_zarr:
                        self.root['meta'].zeros(name=key, shape=new_shape, chunks=new_shape, dtype=np.float32)
                    else:
                        self.root['meta'][key] = np.zeros(new_shape, dtype=np.float32)
                arr = self.root['meta'][key]
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
                arr[new_len-1] = meta_data[key]

            for key in ['beta_priori', 'beta_priori_2']:
                new_shape = (new_len,) + (2,)
                if key not in self.root['meta']:
                    if is_zarr:
                        self.root['meta'].zeros(name=key, shape=new_shape, chunks=new_shape, dtype=np.float32)
                    else: 
                        self.root['meta'][key] = np.zeros(new_shape, dtype=np.float32)
                arr = self.root['meta'][key]
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
                arr[new_len-1] = meta_data[key]

    # ============= Get episodes ===============
    def get_pref_episode(self, idx: int, copy: bool = False):
        """
        Get a pair of episodes by index, including observation and action sequences for both trajectories.
        """
        if copy:
            return {
                'obs': self.root['data']['obs'][idx].copy(),
                'action': self.root['data']['action'][idx].copy(),
                'obs_2': self.root['data']['obs_2'][idx].copy(),
                'action_2': self.root['data']['action_2'][idx].copy(),
                'votes': self.root['meta']['votes'][idx].copy(),
                'votes_2': self.root['meta']['votes_2'][idx].copy(),
                'length': self.root['meta']['length'][idx].copy(),
                'length_2': self.root['meta']['length_2'][idx].copy(),
                'beta_priori': self.root['meta']['beta_priori'][idx].copy(),
                'beta_priori_2': self.root['meta']['beta_priori_2'][idx].copy(),
            }
        else:
            return {
                'obs': self.root['data']['obs'][idx],
                'action': self.root['data']['action'][idx],
                'obs_2': self.root['data']['obs_2'][idx],
                'action_2': self.root['data']['action_2'][idx],
                'votes': self.root['meta']['votes'][idx],
                'votes_2': self.root['meta']['votes_2'][idx],
                'length': self.root['meta']['length'][idx],
                'length_2': self.root['meta']['length_2'][idx],
                'beta_priori': self.root['meta']['beta_priori'][idx],
                'beta_priori_2': self.root['meta']['beta_priori_2'][idx],
            }

    def get_episode_slice(self, idx):
        """
        Get the slice range for an episode based on the index for slicing observation and action arrays.
        """
        return slice(idx, idx + 1)

    # ============= Save methods ===============
    def save_to_store(self, store, chunks: Optional[Dict[str, tuple]] = dict(),
                      compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
                      if_exists='replace', **kwargs):
        """
        Save the replay buffer to the given store with optional chunking and compression.
        """
        root = zarr.group(store=store)
        # Save data and meta with chunking and compression if provided
        for key, value in self.root['data'].items():
            cks = self._resolve_array_chunks(chunks, key, value)
            cpr = self._resolve_array_compressor(compressors, key, value)
            arr = self.root['data'][key]
            zarr.copy(arr, root['data'], name=key, chunks=cks, compressor=cpr)
        
        # Save meta
        for key, value in self.root['meta'].items():
            zarr.copy(value, root['meta'], name=key)
    
    def save_to_path(self, zarr_path, chunks: Optional[Dict[str, tuple]] = dict(),
                     compressors: Union[str, numcodecs.abc.Codec, dict] = dict(), 
                     if_exists='replace', **kwargs):
        """
        Save the replay buffer to a path.
        """
        store = zarr.DirectoryStore(zarr_path)
        return self.save_to_store(store, chunks=chunks, compressors=compressors, if_exists=if_exists, **kwargs)

    # ============= Helper methods ===============
    @classmethod
    def _resolve_array_chunks(cls,
            chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks

    @classmethod
    def _resolve_array_compressor(cls, compressors: Union[dict, str, numcodecs.abc.Codec], key, array):
        # Resolves the compressor for the array
        return compressors.get(key, array.compressor if isinstance(array, zarr.Array) else numcodecs.Blosc())

    @property
    def n_steps(self):
        return len(self.root['meta']['votes'])

    @property
    def data(self):
        return self.root['data']

    @property
    def meta(self):
        return self.root['meta']


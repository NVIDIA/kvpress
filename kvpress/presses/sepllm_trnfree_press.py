# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn
from transformers.cache_utils import Cache

from typing import Any, Dict, List, Optional, Tuple, Union

from kvpress.presses.base_press import BasePress

from transformers import QuantizedCache


class SepCache(Cache):
    """
    A cache as described in the [SepLLM paper - ICML 2025](https://arxiv.org/abs/2412.12094). In the training phase,
    SepLLM condenses the segment information into the KV of the separator that divides the segment. In the inference phase, the
    corresponding SepCache only needs to store the KVs of initial tokens, separator tokens, and recent tokens for generation.

    It stores the Key and Value states as lists of tensors, two lists for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Frequently-Used Parameters:

        `init_cache_size: Union[int, List]`:
            The maximum number of KVs to be stored for initial tokens.
            In the paper, the hyperparameter `a` is an abbreviated alias for `self.init_cache_size`.

        `sep_cache_size: Union[int, List]`:
            The maximum number of KVs to be stored for separator tokens.
            In the paper, the hyperparameter `s` is an abbreviated alias for `self.sep_cache_size`.

        `local_size: Union[int, List]`:
            The maximum number of KVs to be stored for local tokens (i.e., sliding window).
            In the paper, the hyperparameter `w` is an abbreviated alias for `self.local_size`.

        `cache_size: Union[int, List]`:
            The maximum number of KVs to be stored for all the tokens, i.e., the size for the whole KV cache.
            In the paper, the hyperparameter `c` is an abbreviated alias for `self.cache_size`.

        Concerning these four parameters above:
            When a list is passed (its length must be `layer_num`), it represents different values for each layer.
            When an integer is passed, it means the setting is the same for all layers.


        `USE_MAX_SEP_CACHE: bool`:
            If True, it means we only keep at most `self.sep_cache_size` seperators' KVs.
            If the number exceeds this limit, older separator's KVs will be discarded, keeping only the most recent `self.sep_cache_size` KVs.
            In the paper, the hyperparameter `s` is an abbreviated alias for `self.sep_cache_size`.

        `separator_token_ids: List[int]`:
            The token ids of the separator tokens for the current model's tokenizer.
            We have some examples, such as the Llama-3 series models, where setting `model_type='llama'` allows you
                to skip setting `separator_token_ids` and `PADDING_ID` (SepCache will auto-fill them).

        `PADDING_ID: int`:
            The token id of the padding token. You can just set `PADDING_ID` to the id of "<|endoftext|>" token of the tokenizer for the pretrained model.

    Important Note:
        When `cache_size` and `local_size` are set to infinity (i.e., sufficiently large positive integers), and `USE_MAX_SEP_CACHE` is `False`, `SepCache` degenerates into a regular Cache.
        However, you must always ensure that `init_cache_size` + `sep_cache_size` + `local_size` + `left_padding_offset` < `cache_size`.
        Here, `left_padding_offset` denotes the number of padding tokens in the record with the largest left paddings within a runtime batch. `left_padding_offset` can only be determined at runtime.
        To guarantee the above inequality always holds during runtime, when setting, you can intentionally create a sufficient margin between both sides of the following inequality:
            `init_cache_size` + `sep_cache_size` + `local_size`  < `cache_size`, i.e., `a`+`s`+`w`<`c` in the [SepLLM paper - ICML 2025]
            to leave room for `left_padding_offset`.

        Please refer to the `__init__` function's comments for more details on the parameters.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, SepCache
        >>> import torch
        >>> from huggingface_hub import login
        >>> login("hf_xxxXXXxxx")


        >>> def to_cuda(a_dict: dict) -> dict:
        >>>    new_dict = {}
        >>>    for k,v in a_dict.items():
        >>>        if isinstance(v, torch.Tensor):
        >>>            new_dict[k] = v.cuda()
        >>>        else:
        >>>            new_dict[k] = v
        >>>    return new_dict

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", attn_implementation="flash_attention_2", device_map="cuda:0")
        >>> model.bfloat16().cuda()
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        >>> inputs = tokenizer(text="My name is Llama 3", return_tensors="pt")
        >>> inputs = to_cuda(inputs)
        >>> # Prepare a cache and pass it to model's forward; `layer_num` is the number of layers for the pretrained model.
        >>> past_key_values = SepCache(init_cache_size=4, sep_cache_size=128, local_size=256, cache_size=512, layer_num=32, USE_MAX_SEP_CACHE=True, model_type='llama')
        >>> # `separator_token_ids` and `PADDING_ID` must also be provided if you are not using `model_type='llama'` like this demo.
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access SepCache filled with keys/values
        SepCache()
        ```

        ```python
        >>> ## When using the `update` function of SepCache to update the keys/values and the past token ids (necessary in SepCache), the current `input_ids` must also be provided.
        >>> key_states, value_states = past_key_values.update(
                    key_states = key_states,
                    value_states = value_states,
                    input_ids = input_ids,
                    layer_idx = layer_idx,
                    PREFILLING_FLAG = q_len > 1, ## `q_len` is the sequence length of the current `query_states`
                    )

        ```
        For detailed usage instructions, please refer to https://github.com/HKUDS/SepLLM
    """
    # is_sliding = True
    
    @staticmethod
    def slice_on_1d(x, start, end):
        return x[:, start:end, ...]

    @staticmethod
    def slice_on_2d(x, start, end):
        return x[:, :, start:end, ...]

    @staticmethod
    def slice_on_3d(x, start, end):
        return x[:, :, :, start:end, ...]


    @staticmethod
    def sep_1bat_select_on_1d(x, Bid, sep_index, min_sep_num=None, max_sep_num=None, SEP_PADDING_IN_BATCH=True):
        """
        For the record with index `Bid` in a batch, extract the K/V states corresponding to the separators on dimension 1.
           If `SEP_PADDING_IN_BATCH=True`, pad to the longest length (i.e. `max_sep_num`);
           otherwise, truncate to the shortest length (i.e. `min_sep_num`).
        """
        sep_index = sep_index.to(x.device)

        if SEP_PADDING_IN_BATCH:  # Need padding
            assert max_sep_num is not None, f"if `SEP_PADDING_IN_BATCH=True`, `max_sep_num` should not be None"
            new_x_sep = x[Bid, sep_index, ...]  # batch x seqlen x head x dim  -->  sep_num x head x dim
            padding_num = max_sep_num - new_x_sep.shape[0]
            if padding_num > 0:
                assert padding_num <= x.shape[1], f"`padding_num` should be <= `x.shape[1]`, i.e.  x's seqlen"
                new_x_pad = x[Bid, -padding_num:, ...]  # padding_num x head x dim
                return torch.cat([new_x_sep, new_x_pad], dim=0)  # max_sep_num x head x dim
            else:
                return new_x_sep  # max_sep_num x head x dim

        if min_sep_num is None:
            return x[Bid, sep_index, ...]  # batch x seqlen x head x dim -->  sep_num x head x dim
        else:  # `min_sep_num` is provided. Need truncation
            new_x = x[Bid, sep_index, ...]  # batch x seqlen x head x dim -->  sep_num x head x dim
            return new_x[:min_sep_num, ...]  # min_sep_num x head x dim


    @staticmethod
    def sep_1bat_select_on_2d(x, Bid, sep_index, min_sep_num=None, max_sep_num=None, SEP_PADDING_IN_BATCH=True):
        """
        For the record with index `Bid` in a batch, extract the K/V states corresponding to the separators on dimension 2.
           If `SEP_PADDING_IN_BATCH=True`, pad to the longest length (i.e. `max_sep_num`);
           otherwise, truncate to the shortest length (i.e. `min_sep_num`).
        """
        sep_index = sep_index.to(x.device)

        if SEP_PADDING_IN_BATCH:  # Need padding
            assert max_sep_num is not None, f"if `SEP_PADDING_IN_BATCH=True`, `max_sep_num` should not be None"
            new_x_sep = x[Bid, :, sep_index, ...]  # batch x head x seqlen x dim -->  head x sep_num x dim
            padding_num = max_sep_num - new_x_sep.shape[-2]
            if padding_num > 0:
                assert padding_num <= x.shape[-2], f"`padding_num` should be <= `x.shape[-2]`, i.e.  x's seqlen"
                new_x_pad = x[Bid, :, -padding_num:, ...]  # head x padding_num x dim
                return torch.cat([new_x_sep, new_x_pad], dim=-2)  # head x max_sep_num x dim
            else:
                return new_x_sep  # head x max_sep_num x dim

        if min_sep_num is None:
            return x[Bid, :, sep_index, ...]  # batch x head x seqlen x dim -->  head x sep_num x dim
        else:  # `min_sep_num` is provided. Need truncation
            new_x = x[Bid, :, sep_index, ...]  # batch x head x seqlen x dim -->  head x sep_num x dim
            return new_x[:, :min_sep_num, ...]  # head x min_sep_num x dim


    @staticmethod
    def sep_1bat_select_on_3d(x, Bid, sep_index, min_sep_num=None, max_sep_num=None, SEP_PADDING_IN_BATCH=True):
        """
        For the record with index `Bid` in a batch, extract the K/V states corresponding to the separators on dimension 3.
           If `SEP_PADDING_IN_BATCH=True`, pad to the longest length (i.e. `max_sep_num`);
           otherwise, truncate to the shortest length (i.e. `min_sep_num`).
        """
        sep_index = sep_index.to(x.device)

        if SEP_PADDING_IN_BATCH:  # Need padding
            assert max_sep_num is not None, f"if `SEP_PADDING_IN_BATCH=True`, `max_sep_num` should not be None"
            new_x_sep = x[Bid, :, :, sep_index, ...]  # batch x head x dim x seqlen  -->  head x dim x sep_num
            padding_num = max_sep_num - new_x_sep.shape[-1]
            if padding_num > 0:
                assert padding_num <= x.shape[-1], f"`padding_num` should be <= `x.shape[-1]`, i.e.  x's seqlen"
                new_x_pad = x[Bid, :, :, -padding_num:, ...]  # head x dim x padding_num
                return torch.cat([new_x_sep, new_x_pad], dim=-1)  # head x dim x max_sep_num
            else:
                return new_x_sep  # head x dim x max_sep_num

        if min_sep_num is None:
            return x[Bid, :, :, sep_index, ...]  # batch x head x dim x seqlen -->  head x dim x sep_num
        else:  # `min_sep_num` is provided. Need truncation
            new_x = x[Bid, :, :, sep_index, ...]  # batch x head x dim x seqlen -->  head x dim x sep_num
            return new_x[:, :, :min_sep_num, ...]  # head x dim x min_sep_num

    DIM_TO_SLICE = {
        1: slice_on_1d,
        2: slice_on_2d,
        3: slice_on_3d,
    }
    
    BAT_DIM_TO_SELECT = {
        1: sep_1bat_select_on_1d,
        2: sep_1bat_select_on_2d,
        3: sep_1bat_select_on_3d,
    }

    def __init__(self,
                 ## For SepLLM
                 init_cache_size: Union[int, List] = 4,
                 sep_cache_size: Union[int, List] = 64,
                 local_size: Union[int, List] = 256,
                 cache_size: Union[int, List] = 512,
                 SEP_ACCUMULATION: bool = True,
                 USE_MAX_SEP_CACHE: bool = False,
                 SEP_PADDING_IN_BATCH: bool = False,
                 separator_token_ids: List[int] = None,  # required for initialization if `model_type` is not provided.
                 PADDING_ID: int = None,  # required for initialization if `model_type` is not provided.

                 ## For inheritance & initialization states
                 past_tok_ids: List[torch.Tensor] = None,  # It saves all the token ids corresponding to the saved KVs for all layers in SepCache.
                 key_cache: List[torch.Tensor] = None,
                 value_cache: List[torch.Tensor] = None,

                 ## For debugging
                 PRINT_KV_RATIO_INSIDE: bool = False,
                 print_KV_inside_per_steps: int = 1000,
                 _seen_tokens: int = 0,
                 _kept_kv_ratio: List[Tuple[int]] = None,

                 ### For positional encoding shifting
                 APPLY_PE_SHIFT: bool = False,
                 APPLY_PES_INSIDE: bool = True,
                 _shifted_position_ids: List[torch.Tensor] = None,
                 _rope_unsqueeze_dim: int = 1,  # The unsqueeze_dim when applying RoPE.
                 _rope_seq_dim: int = 1,  # The seq_len dimension for the `cos` or `sin` tensors.
                 pe_scaling_factor: float = 1.0,
                 pe_dim: int = 128,  # The number of dims for positional encoding. Typically, just set the `head_dim` to this.
                 max_position_embeddings: int = 8192,
                 base: int = 10000,  # The base for RoPE.

                 ## For basic transformer architecture
                 k_seq_dim: int = 2,  # The dimension for seq_len in key tensors
                 v_seq_dim: int = 2,  # The dimension for seq_len in value tensors
                 layer_num: int = None,  # required for initialization

                 model_type: str = None,  # The model type for running the example. choose from ['llama', 'pythia','falcon'].
                 device = None
                 ) -> None:
        """
        `SEP_ACCUMULATION`: If True, it means we will try to accumulate all the KVs for seperators. If False, only the `new_sep_kv` compressed from the `past_win_kv` will be kept (see function `compress_kv_cache_and_tokids_layer_wise`).

        `USE_MAX_SEP_CACHE`: If True, it means we only keep at most `self.sep_cache_size` seperators' KVs.  If the number exceeds this limit, older separator's KVs will be discarded, keeping only the most recent `self.sep_cache_size` KVs. In the paper, the hyperparameter `s` is an abbreviated alias for `self.sep_cache_size`.

        `SEP_PADDING_IN_BATCH`: If True, it means that SepCache will pad separator tokens in other records to be aligned with the record with the most separators in a batch. If False, it means that SepCache will truncate older separator tokens in other records to be aligned with the record with the fewest separators in a batch.

        Note: If `SEP_ACCUMULATION=True` and `USE_MAX_SEP_CACHE=False`, as the number of input tokens increases, the number of separators in the KV cache will also accumulate endlessly
              and `self.cache_size` will also be infinitely expanded (no longer fixed).

              When `SEP_PADDING_IN_BATCH=True` is used in combination with `USE_MAX_SEP_CACHE=False` and `SEP_ACCUMULATION=True`, the KV cache will accumulate indefinitely,
              and since `SEP_PADDING_IN_BATCH=True`, the KVs of all separators will be retained (rather than being truncated).


        For detailed usage instructions, please refer to https://github.com/HKUDS/SepLLM
        """

        super().__init__()
        if (key_cache is not None) or (value_cache is not None) or (past_tok_ids is not None):
            assert isinstance(key_cache, list)
            assert isinstance(value_cache, list)
            assert isinstance(past_tok_ids, list), f"For SepCache, if `key_cache` and `value_cache` are given (e.g., provided from legacy `past_key_values`), `past_tok_ids` corresponding to `key_cache` and `value_cache` must also be provided to initialize SepCache."

            assert len(key_cache) == len(past_tok_ids), f"The length of `key_cache` ({len(key_cache)}) should be equal to that of `past_tok_ids` ({len(past_tok_ids)})."
            assert len(value_cache) == len(past_tok_ids), f"The length of `value_cache` ({len(value_cache)}) should be equal to that of `past_tok_ids` ({len(past_tok_ids)})."
        assert layer_num is not None, f"`layer_num` must be provided according to the pretrained model."

        ## For basic parameters & states
        self.key_cache: List[torch.Tensor] = key_cache if key_cache is not None else []
        self.value_cache: List[torch.Tensor] = value_cache if value_cache is not None else []    

        self.k_seq_dim = k_seq_dim  # The dimension for the seq_len in key states. Typically, 2.
        self.v_seq_dim = v_seq_dim  # The dimension for the seq_len in value states. Typically, 2.

        self.k_slice = self.DIM_TO_SLICE[k_seq_dim]
        self.v_slice = self.DIM_TO_SLICE[v_seq_dim]

        self.k_bat_dim_select = self.BAT_DIM_TO_SELECT[k_seq_dim]
        self.v_bat_dim_select = self.BAT_DIM_TO_SELECT[v_seq_dim]
        self._seen_tokens: int = _seen_tokens  # Used in `generate` to keep tally of how many tokens the cache has seen as well as performing statistics.
        self.layer_num = layer_num
        self.device = device  # Deprecated


        ## For debugging
        self.PRINT_KV_RATIO_INSIDE = PRINT_KV_RATIO_INSIDE
        self.print_KV_inside_per_steps = print_KV_inside_per_steps
        self._print_kv_ratio_count = 0
        self._kept_kv_ratio: List[Tuple[int]] = _kept_kv_ratio if _kept_kv_ratio is not None else []

        ## For Streaming SepLLM
        self.past_tok_ids: List[torch.Tensor] = past_tok_ids if past_tok_ids is not None else []  # It saves all the token ids corresponding to the saved KVs for all layers in SepCache
        self.left_padding_offset = None
        self._set_layer_wise_attribute("init_cache_size", init_cache_size, layer_num)
        self._set_layer_wise_attribute("local_size", local_size, layer_num)
        self._set_layer_wise_attribute("cache_size", cache_size, layer_num)
        self._set_layer_wise_attribute("sep_cache_size", sep_cache_size, layer_num)
        self._set_layer_wise_attribute("sep_exrange", 0, layer_num)  # runtime right boundary for separators, excluded
        self._set_layer_wise_attribute("max_sep_exidx", self._list_element_add(self.sep_cache_size, self.init_cache_size), layer_num)  # max right boundary for separators, excluded
        self.SEP_ACCUMULATION = SEP_ACCUMULATION
        self.USE_MAX_SEP_CACHE = USE_MAX_SEP_CACHE
        self.SEP_PADDING_IN_BATCH = SEP_PADDING_IN_BATCH


        ### For positional encoding shifting
        self.APPLY_PE_SHIFT = APPLY_PE_SHIFT
        self.APPLY_PES_INSIDE = APPLY_PES_INSIDE

        self.cos_sin_rerotation_cache = {}
        self._cos_cache = None
        self._sin_cache = None
        self._shifted_position_ids: List[torch.Tensor] = _shifted_position_ids if _shifted_position_ids is not None else []
        self._rope_unsqueeze_dim = _rope_unsqueeze_dim
        self._rope_seq_dim = _rope_seq_dim        

        self.pe_dim = pe_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.pe_dim, 2, dtype=torch.int64).float().to(device) / self.pe_dim))
        self.inv_freq = inv_freq
        self.pe_scaling_factor = pe_scaling_factor
        self._sin_cached = None
        self._cos_cached = None

        if model_type is None:
            assert isinstance(separator_token_ids, list), f"`separator_token_ids: List[int]` must be correctly provided for initialization unless `model_type` is properly given, which will auto-fiil `separator_token_ids`."
            assert len(separator_token_ids) > 0, f"`separator_token_ids: List[int]` should NOT be empty."
            for i in range(len(separator_token_ids)):
                assert isinstance(separator_token_ids[i], int), f"The ids in `separator_token_ids` must be of the type `int`."
            assert isinstance(PADDING_ID, int), f"`PADDING_ID: int` must be correctly provided for initialization unless `model_type` is properly given, which will auto-fiil `PADDING_ID`."
            self.separator_token_ids = separator_token_ids
            self.PADDING_ID = PADDING_ID
        else:
            assert isinstance(model_type, str), f"`model_type` should be a `str` or `None`."
            if 'llama' in model_type.lower():
                # print("Debug: For Llama's default separators")
                self.separator_token_ids = [128000, 13, 11, 30, 0, 26, 25, 198, 220, 662, 1174, 949, 758, 2652, 551, 720, 256, 262]  # llama3 8b
                self.PADDING_ID = 128009
            elif ('pythia' in model_type.lower()) or ('gpt_neox' in model_type.lower()):
                # print("Debug: For GPTNeox's default separators")
                self.separator_token_ids = [15, 13, 32, 2, 28, 27, 209, 186, 187,    964, 1157, 3736, 2195, 3706, 1163, 2490,  50276,    586, 4928, 50275 ]       # pythia 14b
                self.PADDING_ID = 0
            elif 'falcon' in model_type.lower():
                # print(f"Debug: For Falcon's default separators")
                self.separator_token_ids = [25, 23,  42, 12, 38, 37, 193,  4610,  204, 258, 1212, 23787, 466 ]       # falcon-40b
                self.PADDING_ID = 11
            else:
                raise NotImplementedError(f"NOT implemented for the tokenizer of the backbone model type: `{model_type}`. You must provide `separator_token_ids: List[int]` and `PADDING_ID: int` for initialization in this case! ")
        
        if APPLY_PE_SHIFT:
            print(">>>>>>>>---------#####################################################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                                                     -----------<<<<<<<<")
            print(">>>>>>>>---------  Warning: When `APPLY_PE_SHIFT=True`, SepCache must store the key/value states       ----------<<<<<<<<")
            print(">>>>>>>>---------              before applying positional encoding (specifically RoPE)                -----------<<<<<<<<")
            print(">>>>>>>>---------#####################################################################################-----------<<<<<<<<")
                
        if APPLY_PES_INSIDE:
            print(">>>>>>>>---------#####################################################################################-----------<<<<<<<<")
            print(">>>>>>>>---------                                                                                     -----------<<<<<<<<")
            print(">>>>>>>>---------  Warning: When `APPLY_PES_INSIDE=True`, there is no need to apply rotary positional embedding--<<<<<<<<")
            print(">>>>>>>>---------  within the self_attention function, as this operation will be handled inside the `update`  ---<<<<<<<<")
            print(">>>>>>>>---------  function of SepCache. Note that `APPLY_PES_INSIDE=True` is typically used together with     ---<<<<<<<<")
            print(">>>>>>>>---------  `APPLY_PE_SHIFT=True`.                                                                     ---<<<<<<<<")
            print(">>>>>>>>---------#####################################################################################-----------<<<<<<<<")                            
            

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,        
        layer_idx: int,        
        input_ids: torch.Tensor = None,
        PREFILLING_FLAG: bool = True,
        query_states: Optional[torch.Tensor] = None,        
        position_ids: Optional[torch.Tensor]=None,                
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:        
            `key_states` (`torch.Tensor`):
                The new key states to cache.
            `value_states` (`torch.Tensor`):
                The new value states to cache.
            `input_ids` (`torch.Tensor`)
                The ids of the input tokens (context tokens or autoregressive tokens)                
            `layer_idx` (`int`):
                The index of the layer to cache the states for.
            `PREFILLING_FLAG` (`bool`)
                It should be `True` at pre-filling phase and `False` when decoding

            `query_states` (`Optional[torch.Tensor]`)
                The query states that need positional encoding shifting. Only useful when `self.APPLY_PE_SHIFT=True`
            `position_ids` (`Optional[torch.Tensor]`)
                The original positional ids of the tokens in the input sequence (i.e., indices of positions of each input sequence tokens in the position embeddings)
                Only useful when `self.APPLY_PE_SHIFT=True`, i.e., SepCache will utilize `position_ids` to calculate positional shifting.
            `cache_kwargs` (`Dict[str, Any]`, optional):
                Additional arguments for the cache update. The following arguments can be used in `SepCache`: `sin`,
                `cos`, `sin_q`, `cos_q`, `shifted_pos_ids` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted. (These are only useful when `self.APPLY_PE_SHIFT=True`)

                Only useful when `self.APPLY_PE_SHIFT=True` and `self.APPLY_PES_INSIDE=False`:
                    `cos` and `sin` are the shifted rotation matrices for key states
                    `cos_q` and `sin_q` are the shifted rotation matrices for query states
                    `shifted_pos_ids` is the shifted positional ids for key states
                    
                When `self.APPLY_PE_SHIFT=True` and `self.APPLY_PES_INSIDE=True`:
                    SepCache will utilize `position_ids` to calculate positional shifting.
                
                `partial_rotation_size` means that `partial_rotation_size` slices along certain dimension need to be shifted (i.e., [0, 1, ..., `partial_rotation_size-1`] slices along certain dimension)

        Return:
            A tuple containing the updated key, value, and query states (query states are optional: only applicable when `self.APPLY_PE_SHIFT=True`).

        For detailed usage instructions, please refer to https://github.com/HKUDS/SepLLM
        """

        APPLY_PE_SHIFT = self.APPLY_PE_SHIFT
        APPLY_PES_INSIDE = self.APPLY_PES_INSIDE
        SEP_ACCUMULATION = self.SEP_ACCUMULATION
        USE_MAX_SEP_CACHE = self.USE_MAX_SEP_CACHE
        SEP_PADDING_IN_BATCH = self.SEP_PADDING_IN_BATCH
        
        if input_ids is None:
            input_ids = cache_kwargs.get("input_ids", None)
        assert input_ids is not None, f"`input_ids` must be properly provided when calling `update()` in `SepCache`."

        assert (self.APPLY_PE_SHIFT and (query_states is not None)) or not APPLY_PE_SHIFT, f"If `APPLY_PE_SHIFT=True`, `query_states` should be provided and it will be updated and returned"
                
        # Update the number of seen tokens
        if layer_idx == 0:
            assert key_states.shape[-2] == input_ids.shape[-1], f"`key_states.shape[-2]` ({key_states.shape[-2]}) should be equal to `input_ids.shape[-1]` ({input_ids.shape[-1]})."
            self._seen_tokens += input_ids.shape[-1]

        # [bsz, num_heads, seq_len, head_dim]
        new_kv_pair = (key_states, value_states)
                
        if (key_states.shape[self.k_seq_dim] + self.get_usable_length(layer_idx) < self.cache_size[layer_idx]) or PREFILLING_FLAG:  ## For prefilling
            assert  (PREFILLING_FLAG and key_states.shape[self.k_seq_dim] >= 1)  or (not PREFILLING_FLAG and key_states.shape[self.k_seq_dim] == 1)

            # Update cache and past token ids                
            self.update_kv_cache_and_past_tok_ids(new_kv_pair, input_ids, layer_idx, COMPRESS_KV=False, SEP_ACCUMULATION=SEP_ACCUMULATION, USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE, SEP_PADDING_IN_BATCH=SEP_PADDING_IN_BATCH)
            
            if APPLY_PE_SHIFT:                     
                shifted_keys, shifted_queries = self.apply_shifted_pos_emb(layer_idx, APPLY_PES_INSIDE, PREFILLING_FLAG, key_states, query_states, position_ids, cache_kwargs ) 
                query_states  = shifted_queries
                self.set_kv_cache( (shifted_keys, self.value_cache[layer_idx]), layer_idx)
            
            if PREFILLING_FLAG and layer_idx == 0:
                self.left_padding_offset = self.get_initial_pos_offset(layer_idx)

            ## Count KV usage
            kv_len_ori = self.get_seq_length(layer_idx)
            kv_len_cmp = self.get_usable_length(layer_idx)
            self._update_kv_ratio(kv_len_cmp=kv_len_cmp, kv_len_ori=kv_len_ori, layer_idx=layer_idx)

        else:
            ## Update the KV cache, count KV usage, and compress the KV cache if necessary                        
            kv_len_ori = self.get_seq_length(layer_idx)
            offset_init_size_layer = self.update_kv_cache_and_past_tok_ids(new_kv_pair, input_ids, layer_idx, COMPRESS_KV=True, SEP_ACCUMULATION=SEP_ACCUMULATION, USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE, SEP_PADDING_IN_BATCH=SEP_PADDING_IN_BATCH)
            kv_len_cmp = self.get_usable_length(layer_idx)
            self._update_kv_ratio(kv_len_cmp=kv_len_cmp, kv_len_ori=kv_len_ori, layer_idx=layer_idx)
                        
            if APPLY_PE_SHIFT:                
                shifted_keys, shifted_queries = self.apply_shifted_pos_emb(layer_idx, APPLY_PES_INSIDE, PREFILLING_FLAG, key_states, query_states, position_ids, cache_kwargs )                 
                query_states  = shifted_queries
                self.set_kv_cache( (shifted_keys, self.value_cache[layer_idx]), layer_idx)
            
        if self.PRINT_KV_RATIO_INSIDE:    
            self._print_kv_ratio(layer_idx)

        if query_states is not None:
            return self.key_cache[layer_idx], self.value_cache[layer_idx], query_states
        else:
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
            
    
    def update_kv_cache_and_past_tok_ids(self, new_kv_pair: Tuple[torch.Tensor], input_ids: torch.Tensor, layer_idx: int, COMPRESS_KV=False, SEP_ACCUMULATION:bool=True, USE_MAX_SEP_CACHE:bool=False, SEP_PADDING_IN_BATCH:bool=True) -> None:
        """Update the KV cache and past token ids; compress the KV cache if necessary."""
        assert layer_idx is not None, f"`layer_idx` must be given"
        assert len(new_kv_pair) == 2, f"The length of `new_kv_pair` must be 2."
        assert len(self.key_cache) == len(self.value_cache), f"The layer numbers of stored `self.key_cache` and `self.value_cache` must be the same."

        self.append_past_tok_ids(input_ids, layer_idx)

        key, value = new_kv_pair
                
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key)                        
            self.value_cache.append(value)
            assert len(self.key_cache) - 1  == layer_idx, f"The key_cache should be updated sequentially according to the layer numbering."              
            assert len(self.value_cache) - 1  == layer_idx, f"The value_cache should be updated sequentially according to the layer numbering."      
        else:            
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx] , key], dim=self.k_seq_dim)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx] , value], dim=self.v_seq_dim)

        assert len(self.key_cache) == len(self.value_cache), f"The layer numbers of stored key_cache and value_cache must be the same."
        assert self.key_cache[layer_idx].shape[self.k_seq_dim] == self.value_cache[layer_idx].shape[self.v_seq_dim], "The seq length for key_cache and value_cache must be the same."

        if COMPRESS_KV:
            cmp_past_kv_pairs, cmp_past_tok_ids, offset_init_size_layer = self.compress_kv_cache_and_tokids_layer_wise((self.key_cache[layer_idx], self.value_cache[layer_idx]), layer_idx ,SEP_ACCUMULATION=SEP_ACCUMULATION, USE_MAX_SEP_CACHE=USE_MAX_SEP_CACHE, SEP_PADDING_IN_BATCH=SEP_PADDING_IN_BATCH )
            self.set_kv_cache(cmp_past_kv_pairs, layer_idx)
            self.set_past_tok_ids(cmp_past_tok_ids, layer_idx)            
            return offset_init_size_layer
        

    def append_past_tok_ids(self, input_ids: torch.Tensor, layer_idx: int) -> None:
        """Naively append the new `input_ids` to `self.past_tok_ids[layer_idx]`"""    
        assert layer_idx is not None, f"`layer_idx` must be given"
        
        if len(self.past_tok_ids) <= layer_idx:                        
            self.past_tok_ids.append(input_ids)
            assert len(self.past_tok_ids) - 1  == layer_idx, f"The past_tok_ids should be updated sequentially according to the layer numbering."                        
        else:      
            self.past_tok_ids[layer_idx] = torch.cat([self.past_tok_ids[layer_idx] , input_ids], dim=-1)


    def compress_kv_cache_and_tokids_layer_wise(self, past_kv_pairs, layer_idx:int ,SEP_ACCUMULATION=False, USE_MAX_SEP_CACHE=False, SEP_PADDING_IN_BATCH=True ):
        """        
        `SEP_ACCUMULATION`: If True, it means we will try to accumulate all the KVs for seperators. If False, only the `new_sep_kv` compressed from the `past_win_kv` will be kept (see function `compress_kv_cache_and_tokids_layer_wise`).
                                                             
        `USE_MAX_SEP_CACHE`: If True, it means we only keep at most `self.sep_cache_size` seperators' KVs.  If the number exceeds this limit, older separator's KVs will be discarded, keeping only the most recent `self.sep_cache_size` KVs. In the paper, the hyperparameter `s` is an abbreviated alias for `self.sep_cache_size`.

        `SEP_PADDING_IN_BATCH`: If True, it means that SepCache will pad separator tokens in other records to be aligned with the record with the most separators in a batch. If False, it means that SepCache will truncate older separator tokens in other records to be aligned with the record with the fewest separators in a batch.
        
        Note: If `SEP_ACCUMULATION=True` and `USE_MAX_SEP_CACHE=False`, as the number of input tokens increases, the number of separators in the KV cache will also accumulate endlessly 
              and `self.cache_size` will also be infinitely expanded (no longer fixed).

              When `SEP_PADDING_IN_BATCH=True` is used in combination with `USE_MAX_SEP_CACHE=False` and `SEP_ACCUMULATION=True`, the KV cache will accumulate indefinitely, 
              and since `SEP_PADDING_IN_BATCH=True`, the KVs of all separators will be retained (rather than being truncated).


        For detailed usage instructions, please refer to https://github.com/HKUDS/SepLLM
        """    

        key, value = past_kv_pairs
        seq_len = key.size(self.k_seq_dim)
        assert seq_len == self.get_usable_length(layer_idx), f"The seq_len of cached past key and value states should be the same as the return of `get_usable_length()`, which is {self.get_usable_length(layer_idx)}"

        
        left_padding_offset =  self.left_padding_offset        
        assert left_padding_offset is not None
        offset_init_size_layer = self.init_cache_size[layer_idx] + left_padding_offset
        self._set_layer_wise_attribute("max_sep_exidx", self._list_element_add(self.sep_cache_size, self.init_cache_size, bias=left_padding_offset), self.layer_num)
        self._CHECK_PARAMS_VALIDITY(layer_idx, left_padding_offset)

        if self.sep_exrange[layer_idx] <=0:            
            self.sep_exrange[layer_idx] = offset_init_size_layer

        assert seq_len - self.local_size[layer_idx] > self.sep_exrange[layer_idx]
        
        if offset_init_size_layer > 0:                                                       
            initial_kv, initial_tokids =  self.slice_kv_cache_and_tokids( past_kv_pairs, self.past_tok_ids[layer_idx], 0, offset_init_size_layer, seq_len=seq_len, _CHECK_IDX=True )        

        Before_First_Time_Compress_Flag = (self.sep_exrange[layer_idx] == offset_init_size_layer)  ## If true, it means the present timestamp is before t1: the 1st time to compress the past window, in which only seperators' kv are kept.
        
        if SEP_ACCUMULATION and not Before_First_Time_Compress_Flag: ## To get the old sep kv and sep token ids.           
            past_sep_kv, past_sep_tokids =  self.slice_kv_cache_and_tokids( past_kv_pairs, self.past_tok_ids[layer_idx], offset_init_size_layer, self.sep_exrange[layer_idx], seq_len=seq_len, _CHECK_IDX=True )            
        
        past_win_kv, past_win_tokids =  self.slice_kv_cache_and_tokids( past_kv_pairs, self.past_tok_ids[layer_idx], self.sep_exrange[layer_idx], seq_len - self.local_size[layer_idx], seq_len=seq_len, _CHECK_IDX=True )        
        
        
        local_kv, local_tokids  =  self.slice_kv_cache_and_tokids( past_kv_pairs, self.past_tok_ids[layer_idx], seq_len - self.local_size[layer_idx], seq_len, seq_len=seq_len, _CHECK_IDX=True )
        
        new_sep_kv, new_sep_tokids, min_sep_num, max_sep_num = self.compress_past_win_2_seps( past_win_kv, past_win_tokids, SEP_PADDING_IN_BATCH = SEP_PADDING_IN_BATCH ) ## To get the new sep kv and sep token ids that were just compressed from the past window
        
        if SEP_ACCUMULATION and not Before_First_Time_Compress_Flag:  ## Try to accumulate all the seen seps           
            sep_kv, sep_tokids  = self.cat_kv_cache_and_tokids( [ past_sep_kv, new_sep_kv ] ,  [past_sep_tokids, new_sep_tokids ] )                
            new_sep_len = new_sep_tokids.shape[-1]
            sep_len = sep_tokids.shape[-1]  
        else: ## Only keep the newly obtained kv (those just compressed from the past window)
            sep_kv, sep_tokids = new_sep_kv, new_sep_tokids
            # new_sep_len = new_sep_tokids.shape[-1]
            sep_len = sep_tokids.shape[-1]            
            assert (SEP_PADDING_IN_BATCH and max_sep_num==sep_len) or ( (not SEP_PADDING_IN_BATCH) and min_sep_num==sep_len)


        if USE_MAX_SEP_CACHE: ## Fixed sep cache size, i.e., only keep max_sep_len seps' kv in the cache. 
            if offset_init_size_layer + sep_len > self.max_sep_exidx[layer_idx]:
                max_sep_len = self.max_sep_exidx[layer_idx] - offset_init_size_layer
                assert sep_kv[0].shape[-2] == sep_tokids.shape[-1], f"The seq_len for seps' KVs and tok_ids should be the same."

                sep_kv, sep_tokids =  self.slice_kv_cache_and_tokids( sep_kv, sep_tokids, sep_len-max_sep_len, sep_len, seq_len = sep_tokids.shape[-1] ,_CHECK_IDX=True )
                self.sep_exrange[layer_idx] =  self.max_sep_exidx[layer_idx]  
            else:
                self.sep_exrange[layer_idx] =  offset_init_size_layer + sep_len             

        else:    ## Extend the sep cache and the whole cache if USE_MAX_SEP_CACHE is not set                           
            self.sep_exrange[layer_idx] =  offset_init_size_layer + sep_len
            if self.sep_exrange[layer_idx] > self.max_sep_exidx[layer_idx]:                    
                cache_incremental_gap = self.sep_exrange[layer_idx] - self.max_sep_exidx[layer_idx]
                self.max_sep_exidx[layer_idx] = self.sep_exrange[layer_idx] 
                self.sep_cache_size[layer_idx] = self.sep_cache_size[layer_idx] + cache_incremental_gap
                self.cache_size[layer_idx] = self.cache_size[layer_idx] + cache_incremental_gap

        if offset_init_size_layer > 0:                                
            cmp_past_kv_pairs, cmp_past_tok_ids  = self.cat_kv_cache_and_tokids( [initial_kv, sep_kv, local_kv ] ,  [initial_tokids, sep_tokids, local_tokids  ] )
        else:
            cmp_past_kv_pairs, cmp_past_tok_ids  = self.cat_kv_cache_and_tokids( [sep_kv, local_kv ] ,  [sep_tokids, local_tokids  ] )
                
        return cmp_past_kv_pairs, cmp_past_tok_ids, offset_init_size_layer
            

    def compress_past_win_2_seps(self, past_win_kv: Tuple[torch.Tensor], past_win_tokids: torch.Tensor, MIN_SEP_ALERT: bool=False, SEP_PADDING_IN_BATCH: bool=True ) -> Tuple[Union[Tuple[torch.Tensor], torch.Tensor, int ]]:
        """Compress the KVs in the past window into the sep cache where only separators' KVs are kept. Padding or Truncating if necessary."""
        sep_index_tensor = torch.zeros_like(past_win_tokids).bool()  # batch x seq_len

        for sp_id in self.separator_token_ids:            
            sep_index_tensor = sep_index_tensor | ( past_win_tokids == sp_id ) # batch x seq_len

        sep_cnt = sep_index_tensor.int().sum(-1)
        min_sep_num = sep_cnt.min()  # the min sep number for the seqs in a batch
        max_sep_num = sep_cnt.max()  # the max sep number for the seqs in a batch

        
        if MIN_SEP_ALERT and not SEP_PADDING_IN_BATCH:
            assert min_sep_num>0, f"The min sep number for each compressing time in a batch should be at least one if `MIN_SEP_ALERT=True` and `SEP_PADDING_IN_BATCH=False`"
                
        batch1_sep_ids_list = []
        batch_size = past_win_tokids.shape[0]
        for b_id in range(batch_size):            
            batch1_sep_ids = past_win_tokids[b_id, sep_index_tensor[b_id]] # #  sep_num
            if SEP_PADDING_IN_BATCH: ## padding
                sep_num = batch1_sep_ids.shape[-1]
                padding_num =  max_sep_num - sep_num                       
                if padding_num > 0:
                    assert padding_num <= past_win_tokids.shape[-1], f"padding_num: {padding_num} should be <= past_win_tokids.shape[-1]:{past_win_tokids.shape[-1]}"
                    batch1_sep_ids = batch1_sep_ids  # #  sep_num
                    batch1_pad_ids = past_win_tokids[b_id, -padding_num:]  # #  padding_num
                    batch1_sep_ids =  torch.cat([batch1_sep_ids, batch1_pad_ids], dim =-1)   ##  max_sep_num                
            else: ## truncating
                batch1_sep_ids = batch1_sep_ids[..., :min_sep_num ]  # #  min_sep_num
            batch1_sep_ids_list.append(batch1_sep_ids)                                                           
            
        new_sep_tokids = torch.stack(batch1_sep_ids_list, dim=0) # #  B x min_sep_num
        key_cache, value_cache = past_win_kv

        assert batch_size==key_cache.shape[0]
        batch1_sep_k_list = []
        batch1_sep_v_list = []
        batch1_sep_ids_list = []
        for b_id in range(batch_size):
            batch1_sep_k = self.k_bat_dim_select(key_cache, b_id, sep_index_tensor[b_id], min_sep_num, max_sep_num, SEP_PADDING_IN_BATCH)
            batch1_sep_k_list.append(batch1_sep_k)

            batch1_sep_v = self.v_bat_dim_select(value_cache, b_id, sep_index_tensor[b_id], min_sep_num, max_sep_num, SEP_PADDING_IN_BATCH)
            batch1_sep_v_list.append( batch1_sep_v )   
        
        sep_k = torch.stack(batch1_sep_k_list, dim=0)  ## batch x head x min_sep_num x dim
        sep_v = torch.stack(batch1_sep_v_list, dim=0)  ## batch x head x min_sep_num x dim                   
        new_sep_kv = (sep_k, sep_v)

        return new_sep_kv, new_sep_tokids, min_sep_num, max_sep_num      


    def apply_shifted_pos_emb(self, layer_idx: int, APPLY_PES_INSIDE: bool, PREFILLING_FLAG: bool, key_states: torch.Tensor, query_states: torch.Tensor, position_ids: torch.Tensor, cache_kwargs: Optional[Dict[str, Any]] = None ) -> torch.Tensor:        
        """Perform positional encoding shifting if required"""
        seq_len = self.get_usable_length(layer_idx)
        keys_to_shift = self.key_cache[layer_idx]
        queries_to_shift = query_states
        assert keys_to_shift.shape[self.k_seq_dim] == seq_len
        
        if cache_kwargs is None:
            cache_kwargs = {}

        if APPLY_PES_INSIDE:           
            if len(self._shifted_position_ids) <= layer_idx:
                self._shifted_position_ids.append(None)

            if PREFILLING_FLAG: ## for prefilling
                assert position_ids.shape[-1] >= seq_len, f"The length of position_ids should be >= the usable length of kv cache when prefilling."                
                self._shifted_position_ids[layer_idx] = position_ids[:, :seq_len].detach()
                shifted_pos_ids = self._shifted_position_ids[layer_idx]

            elif self._shifted_position_ids[layer_idx].shape[-1] >= seq_len:  ## for generation
                assert position_ids.shape[-1] == 1, f"The length of query and position_ids should be 1 during generation."
                shifted_pos_ids = self._shifted_position_ids[layer_idx][:, :seq_len].detach()

            elif self._shifted_position_ids[layer_idx].shape[-1] < seq_len:   ## for generation
                assert position_ids.shape[-1] == 1, f"The length of query and position_ids should be 1 during generation."
                increased_gap = seq_len - self._shifted_position_ids[layer_idx].shape[-1]
                assert increased_gap < self._shifted_position_ids[layer_idx].shape[-1], f"Normally, for auto-regressive model, the input length for each step should be 1 during generation."

                new_position_ids = self._shifted_position_ids[layer_idx][:, -increased_gap: ] + increased_gap
                self._shifted_position_ids[layer_idx] = torch.cat([self._shifted_position_ids[layer_idx], new_position_ids.detach()], dim=-1)
                shifted_pos_ids = self._shifted_position_ids[layer_idx]
            else:
                raise RuntimeError

            cos, sin = self._get_naive_shifted_cos_sin(
                key_states, shifted_pos_ids, seq_len
            )

            q_rope_idx = torch.arange( seq_len - query_states.shape[self.k_seq_dim],  seq_len).to(cos.device)
            cos_q, sin_q = cos.index_select(self._rope_seq_dim, q_rope_idx), sin.index_select(self._rope_seq_dim, q_rope_idx)

        else:
            sin = cache_kwargs.get("sin")
            cos = cache_kwargs.get("cos")                         
            sin_q = cache_kwargs.get("sin_q")
            cos_q = cache_kwargs.get("cos_q")    
            shifted_pos_ids = cache_kwargs.get("shifted_pos_ids") 
            assert (sin is not None) and (cos is not None), f"sin and cos matrices should be be provided"
            if sin_q is None:
                q_rope_idx = torch.arange( seq_len - query_states.shape[self.k_seq_dim],  seq_len).to(sin.device)
                sin_q = sin.index_select(self._rope_seq_dim, q_rope_idx)
            if cos_q is None:
                q_rope_idx = torch.arange( seq_len - query_states.shape[self.k_seq_dim],  seq_len).to(cos.device)
                cos_q = cos.index_select(self._rope_seq_dim, q_rope_idx)
            
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        
        # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
        if partial_rotation_size is not None:
            keys_to_shift, keys_pass = (
                keys_to_shift[..., :partial_rotation_size],
                keys_to_shift[..., partial_rotation_size:]
            )
            queries_to_shift, queries_pass = (
                queries_to_shift[..., :partial_rotation_size],
                queries_to_shift[..., partial_rotation_size:]
            )
                                    
        shifted_keys = self._apply_rotary_pos_emb_single(keys_to_shift, cos, sin, shifted_pos_ids, unsqueeze_dim=self._rope_unsqueeze_dim)
        shifted_queries = self._apply_rotary_pos_emb_single(queries_to_shift, cos_q, sin_q, shifted_pos_ids[:,  -queries_to_shift.shape[self.k_seq_dim] : ], unsqueeze_dim=self._rope_unsqueeze_dim)

        if partial_rotation_size is not None:
            shifted_keys = torch.cat( [shifted_keys, keys_pass], dim=-1)
            shifted_queries = torch.cat( [shifted_queries, queries_pass], dim=-1)


        return shifted_keys, shifted_queries


    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the seen tokens. A layer index can be optionally passed."""                
        return self._seen_tokens


    def get_usable_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the actual cached states. A layer index must be passed."""                         
        if len(self.key_cache) <= layer_idx :
            return 0
        assert self.key_cache[layer_idx].shape[self.k_seq_dim] == self.value_cache[layer_idx].shape[self.v_seq_dim], f"`self.key_cache` and `self.value_cache` should have the same length."        
        return self.key_cache[layer_idx].shape[self.k_seq_dim]

    def get_initial_pos_offset(self, layer_idx:int = 0) -> int:      
        """Return the number of padding tokens in the record with the most left padding tokens in a batch."""
        assert isinstance(self.PADDING_ID, int), f"`self.PADDING_ID` should be correctly set."
        assert len(self.past_tok_ids) > layer_idx, f"`self.past_tok_ids` for layer {layer_idx} must have been properly set."
                
        past_tok_ids = self.past_tok_ids[layer_idx]
        assert past_tok_ids is not None, f"`past_tok_ids` for layer {layer_idx} should not be None"

        pad_index_tensor = (past_tok_ids == self.PADDING_ID)  ## batch x seq_len
        pad_toks_cnt = pad_index_tensor.int().sum(-1)  ## [batch]
        offset = pad_toks_cnt.max().item()

        return offset

                             
    def get_batch_size(self) -> int:
        """Return the batch size."""
        assert self.key_cache is not None, f"`self.key_cache` should not be None."
        assert self.value_cache is not None, f"`self.value_cache` should not be None."
        assert len(self.key_cache) > 0, f"`self.key_cache` is empty. No batch size is available."
        assert len(self.value_cache) > 0, f"self.value_cache is empty. No batch size is available."

        assert len(self.value_cache) == len(self.key_cache), f"self.value_cache and self.key_cache should be at the same length."
        assert self.value_cache[0].shape[0] == self.key_cache[0].shape[0], f"self.value_cache and self.key_cache should have the same batch size."

        return self.value_cache[0].shape[0]

    def get_kv_pair(self, layer_idx: int = None) -> Tuple[torch.Tensor]:
        assert layer_idx is not None, f"`layer_idx` must be given."

        if (len(self.key_cache) <= layer_idx) and (len(self.value_cache) <= layer_idx ):
            key = self.key_cache[layer_idx]
            value = self.value_cache[layer_idx]
        else:
            raise RuntimeError(f"The KV for layer:{layer_idx} have not been set.")
        return (key, value)


    def set_kv_cache(self, kv_pair: Tuple , layer_idx: int ) -> None:
        assert len(kv_pair) == 2, f"The length of `kv_pair` must be 2."
        self.key_cache[layer_idx] = kv_pair[0]
        self.value_cache[layer_idx] = kv_pair[1]
    
    def set_past_tok_ids(self, tok_ids: torch.Tensor, layer_idx:int) -> None:
        self.past_tok_ids[layer_idx] = tok_ids


    def cat_kv_cache_and_tokids(self, kv_pairs_list: List[Tuple[torch.Tensor]] , tok_ids_list:List[torch.Tensor]) -> Tuple[Union[Tuple[torch.Tensor],torch.Tensor]]:
        
        return self.cat_kv_cache(kv_pairs_list), self.cat_token_ids(tok_ids_list)


    def slice_kv_cache_and_tokids(self, kv_pair:Tuple[torch.Tensor], tok_ids_list:torch.Tensor, start:int, end:int, seq_len:int=None, _CHECK_IDX:bool=True, ) -> Tuple[Union[Tuple[torch.Tensor], torch.Tensor]]:
                             
        sliced_kv = self._slice_kv(start, end,  kv_pair=kv_pair, seq_len=seq_len, _CHECK_IDX=_CHECK_IDX,)                                    
        sliced_tids = self._slice_tok_ids(start, end, tok_ids_list = tok_ids_list, seq_len=seq_len, _CHECK_IDX=_CHECK_IDX)
        
        return sliced_kv , sliced_tids


    def cat_kv_cache(self, kv_pairs_list: List[Tuple[torch.Tensor]] ) -> Tuple[torch.Tensor]:               
        assert len(kv_pairs_list) >= 1 
        
        if len(kv_pairs_list) == 1 :
            return kv_pairs_list[0]
        else:
            ret = None 
            for i, kv_pair in enumerate(kv_pairs_list): # enumerate all the KVs needed to be cat
                if i == 0:
                    ret = kv_pair
                else:
                    ret = self._cat_kv(ret, kv_pair)
            return ret


    def cat_token_ids(self, tok_ids_list:List[torch.Tensor]  ) -> torch.Tensor :
        assert len(tok_ids_list) >= 1 
        
        return torch.cat(tok_ids_list, dim=-1)     


    def _cat_kv(self, kv_pair_a:Tuple[torch.Tensor],  kv_pair_b:Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:            
        k_a, v_a = kv_pair_a
        k_b, v_b = kv_pair_b
        
        cat_k = torch.cat([k_a, k_b], dim=self.k_seq_dim)
        cat_v = torch.cat([v_a, v_b], dim=self.v_seq_dim)
        return (cat_k, cat_v)


    def _slice_kv(self, start:int, end:int, kv_pair: Tuple[torch.Tensor],   seq_len:int=None, _CHECK_IDX:bool=True)  -> Tuple[torch.Tensor] :
        assert kv_pair is not None, f"kv_pair must NOT be None when slicing it."
        key_cache = kv_pair[0]
        value_cache = kv_pair[1]

        if _CHECK_IDX:                                 
            assert seq_len is not None, f"seq_len must be given for checking the index for slicing"
            start, end = self._CHECK_IDX(start, end, seq_len)   
            
        sliced_key_cache = self.k_slice(key_cache, start, end) 
        sliced_value_cache = self.v_slice(value_cache, start, end)

        return ( sliced_key_cache, sliced_value_cache)


    def _slice_tok_ids(self, start:int, end:int, tok_ids_list:torch.Tensor , seq_len:int=None, _CHECK_IDX:bool=False) -> torch.Tensor:
        assert tok_ids_list is not None, f"tok_ids_list must NOT be None when slicing it."
        
        if _CHECK_IDX:
            assert seq_len is not None, f"seq_len must be given for checking the index for slicing"
            start, end = self._CHECK_IDX(start, end, seq_len)        
          
        sliced_tok_ids = tok_ids_list[:, start:end]
        return sliced_tok_ids

    def _set_layer_wise_attribute(self, name: str, value: Any, layer_num:int ):
        """Set layer-wise attributes"""
        if isinstance(value, int):        
            setattr(self, name, [value] * layer_num)
        elif isinstance(value, (list, tuple)):
            assert len(value) == layer_num, f"The length of {name}: {len(value)} must be equal to `layer_num`: {layer_num}"
            setattr(self, name, list(value))
        else:
            raise TypeError(f"{name} must be of the type `int` or `list` but got `{type(value)}`")

    def _list_element_add(self, list_a: List, list_b: List, bias: int=0, dtype = int, device = 'cpu') -> List:  
        """Element-wise addition between two lists."""      
        assert len(list_a) == len(list_b), f"The length of `list_a` ({len(list_a)}) must be equal to that of `list_b` ({len(list_b)})."
        tensor_c = torch.tensor(list_a, dtype=dtype, device=device) + torch.tensor(list_b, dtype=dtype, device=device) + torch.tensor([bias], dtype=dtype, device=device)
        return tensor_c.int().tolist()
        
    def _CHECK_IDX(self, start: int = 0, end: int = 100, seq_len: int = 1000):
        assert isinstance(start, int) and isinstance(end, int) and isinstance(seq_len, int), f"`start`, `end`, `seq_len` must be `int`."
        assert seq_len>0 , f"`seq_len` must > 0"
        
        if start <0 :
            start = start % seq_len
        if end < 0 :
            end = end % seq_len
        assert (start >=0) and (start < seq_len) , f"start:{start}, end:{end}, seq_len:{seq_len}"
        assert (end >= 0) and (end <= seq_len) , f"start:{start}, end:{end}, seq_len:{seq_len}"
        assert  start < end, f"start:{start}, end:{end}, seq_len:{seq_len}"

        return start,end

    def _CHECK_PARAMS_VALIDITY(self, layer_idx:int, left_padding_offset:int):
        assert len(self.cache_size) > layer_idx
        assert len(self.init_cache_size) > layer_idx
        assert len(self.sep_cache_size) > layer_idx
        assert len(self.max_sep_exidx) > layer_idx
        assert len(self.local_size) > layer_idx

        assert self.cache_size[layer_idx] > 0 , f"`self.cache_size` for layer:{layer_idx} must be greater than 0"
        assert self.init_cache_size[layer_idx] >= 0 , f"`self.init_cache_size` for layer:{layer_idx} must be greater than (equal to) 0"
        assert self.local_size[layer_idx] > 0 , f"`self.local_size` for layer:{layer_idx} must be greater than 0"
                    
        assert self.sep_cache_size[layer_idx] > 0 , f"`self.sep_cache_size` for layer:{layer_idx} must be greater than 0"
        assert self.max_sep_exidx[layer_idx] > 0 , f"`self.max_sep_exidx` for layer:{layer_idx} must be greater than 0"
        assert self.init_cache_size[layer_idx] + self.sep_cache_size[layer_idx] + self.local_size[layer_idx] + left_padding_offset < self.cache_size[layer_idx], f"`init_cache_size` ({self.init_cache_size[layer_idx]}) + `sep_cache_size` ({self.sep_cache_size[layer_idx]}) + `local_size` ({self.local_size[layer_idx]}) + `left_padding_offset` ({left_padding_offset}) for layer {layer_idx} should be less than `cache_size`:({self.cache_size[layer_idx]}) for layer {layer_idx}, i.e., a + s + w + (left_padding_offset) < c. Please increase `cache_size` if applicable."
        


    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb_single(self, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """        
        cos = cos.unsqueeze(unsqueeze_dim)   # batch x seq_len x dim  --> batch x 1 x seq_len x dim
        sin = sin.unsqueeze(unsqueeze_dim)        
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return  k_embed


    def _get_naive_shifted_cos_sin(self, x: torch.Tensor, position_ids: torch.Tensor=None, seq_len=None):
        # x: [batch, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        # backwards compatibility
        self._cos_cached = cos
        self._sin_cached = sin
        return cos, sin
    

    def _get_scaled_shifted_cos_sin(self, x, position_ids, seq_len=None):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = self._get_naive_shifted_cos_sin(x, position_ids, seq_len)
        return cos, sin


    def _get_dynamicNTK_scaling_shifted_cos_sin(self, x, position_ids, seq_len=None):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO: this may break with compilation

        cos, sin = self._get_naive_shifted_cos_sin(x, position_ids, seq_len)
        return cos, sin


    def _update_kv_ratio(self, kv_len_cmp:int, kv_len_ori:int, layer_idx: int=0) -> None:
        """Update the KV ratios which are for statistics and debugging."""
        if len(self._kept_kv_ratio) <= layer_idx:
            self._kept_kv_ratio.append( (kv_len_cmp,  kv_len_ori ) )    
        else:
            old_kv_len_cmp = self._kept_kv_ratio[layer_idx][0]
            old_kv_len_ori = self._kept_kv_ratio[layer_idx][1]
            self._kept_kv_ratio[layer_idx] = (old_kv_len_cmp + kv_len_cmp,  old_kv_len_ori + kv_len_ori )
            
    def _print_kv_ratio(self, layer_idx : int, LAYER_WISE: bool = False):
        """Print the KV ratios."""
        self._print_kv_ratio_count += 1 
        if LAYER_WISE:
            if self._print_kv_ratio_count % self.print_KV_inside_per_steps == 0:      
                print(f"######################## [Kept Tokens, Seen Tokens] : {self._kept_kv_ratio[layer_idx]}, Ratio: { (self._kept_kv_ratio[layer_idx][0]+1e-6) / (self._kept_kv_ratio[layer_idx][1]+1e-6) :.4f} ########################")    

        elif self._print_kv_ratio_count % (self.print_KV_inside_per_steps * self.layer_num) == 0:                
            print(f"######################## [Kept Tokens, Seen Tokens] : {self._kept_kv_ratio[layer_idx]}, Ratio: { (self._kept_kv_ratio[layer_idx][0]+1e-6) / (self._kept_kv_ratio[layer_idx][1]+1e-6) :.4f} ########################")    


    @classmethod ## Deprecated
    def from_legacy_cache(cls, 
                past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,

                ## For SepLLM                                
                init_cache_size: Union[int, List] = 4,        
                sep_cache_size: Union[int, List] = 64,
                local_size: Union[int, List]=256, 
                cache_size: Union[int, List]=512,    
                SEP_ACCUMULATION: bool = True,
                USE_MAX_SEP_CACHE: bool = False,
                SEP_PADDING_IN_BATCH: bool = False,
                separator_token_ids: List[int] = None, ## required for initialization if `model_type` is not provided. set it to `[-1]` to degrade SepCache to StreamingLLM's SinkCache
                PADDING_ID: int = None, ## required for initialization if `model_type` is not provided.

                ## For inheritance & initialization states
                past_tok_ids: List[torch.Tensor] = None,  ## It saves all the token ids corresponding to the saved KVs for all layers in SepCache.                
                key_cache: List[torch.Tensor] = None,          
                value_cache: List[torch.Tensor] = None,

                ## For debugging
                PRINT_KV_RATIO_INSIDE: bool = False,
                print_KV_inside_per_steps: int = 1000,   
                _seen_tokens: int = 0, 
                _kept_kv_ratio: List[Tuple[int]] = None,
                
                ### For positional encoding shifting
                APPLY_PE_SHIFT: bool = False,
                APPLY_PES_INSIDE: bool = True,
                _shifted_position_ids:  List[torch.Tensor] = None,
                _rope_unsqueeze_dim: int = 1, ## The unsqueeze_dim when applying RoPE.
                _rope_seq_dim: int=1, ## The seq_len dimension for the `cos` or `sin` tensors.
                pe_scaling_factor:float = 1.0,
                pe_dim:int=128, ## The number of dims for positional encoding. Typically, just set the `head_dim` to this.
                max_position_embeddings: int = 8192, 
                base: int=10000,  ## The base for RoPE.               
                
                ## For basic transformer architecture
                k_seq_dim: int=2, ## The dimension for seq_len in key tensors
                v_seq_dim: int=2, ## The dimension for seq_len in value tensors
                layer_num: int = None, ## required for initialization

                model_type: str = None,  ## The model type for running the example. choose from ['llama', 'pythia','falcon'].
                device = None    
    ) -> "SepCache":
        """Deprecated: Converts a cache in the legacy cache format into `SepCache`."""   

        if past_key_values is not None:
            assert len(past_key_values)==0, f"`from_legacy_cache` function is deprecated. You can only use it when `past_key_values=None` or `past_key_values` is empty, in which case, `from_legacy_cache` is equivalent to the `__init__` function."        
            past_key_values = None

        assert past_key_values is None, f"`from_legacy_cache` function is deprecated. You can only use it when `past_key_values=None` or `past_key_values` is empty, in which case, `from_legacy_cache` is equivalent to the `__init__` function."        
        

        if past_key_values is not None: ## Deprecated
            key_cache = []
            value_cache = []               
            
            for i, kv in enumerate(past_key_values):
                if i == 0:
                    past_tok_ids = [] if len(kv) == 4  else past_tok_ids       

                if len(kv) == 4:
                    k, v, p_tok_ids, _seen_tokens  = kv
                    key_cache.append(k)
                    value_cache.append(v)
                    past_tok_ids.append(p_tok_ids)
                    _seen_tokens = _seen_tokens
                elif len(kv) == 2:
                    k, v = kv
                    key_cache.append(k)
                    value_cache.append(v)
                    
        cache = cls(
                ## For SepLLM                
                init_cache_size = init_cache_size,        
                sep_cache_size = sep_cache_size,
                local_size = local_size, 
                cache_size = cache_size,                    
                SEP_ACCUMULATION = SEP_ACCUMULATION,
                USE_MAX_SEP_CACHE = USE_MAX_SEP_CACHE,
                SEP_PADDING_IN_BATCH = SEP_PADDING_IN_BATCH,
                separator_token_ids = separator_token_ids,
                PADDING_ID = PADDING_ID,

                ## For inheritance & initialization states
                past_tok_ids = past_tok_ids,  ## It saves all the token ids corresponding to the saved KVs for all layers in SepCache        
                key_cache = key_cache,          
                value_cache = value_cache,

                ## For debugging
                PRINT_KV_RATIO_INSIDE = PRINT_KV_RATIO_INSIDE,
                print_KV_inside_per_steps = print_KV_inside_per_steps,   
                _seen_tokens = _seen_tokens, 
                _kept_kv_ratio = _kept_kv_ratio,
                
                ### For positional encoding shifting
                APPLY_PE_SHIFT = APPLY_PE_SHIFT,
                APPLY_PES_INSIDE = APPLY_PES_INSIDE,
                _shifted_position_ids = _shifted_position_ids,
                _rope_unsqueeze_dim = _rope_unsqueeze_dim,
                _rope_seq_dim = _rope_seq_dim, 
                pe_scaling_factor = pe_scaling_factor,
                pe_dim = pe_dim,
                max_position_embeddings = max_position_embeddings, 
                base = base,                 
                
                ## For basic transformer architecture
                k_seq_dim = k_seq_dim,
                v_seq_dim = v_seq_dim,
                layer_num = layer_num,
                
                model_type = model_type,  
                device = device,   
        )

        return cache

    
    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]: ## Deprecated
        """Deprecated: Converts the `SepCache` instance into the legacy cache format, i.e., tuple."""
        print(">>>>>>>>>>>Warnings: Please try to avoid using this deprecated `to_legacy_cache` function since it will drop many useful parameters or states in SepCache.<<<<<<<<<<<")
        legacy_cache = ()
        for layer_idx in range(len(self.key_cache)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx], self.past_tok_ids[layer_idx], self._seen_tokens), )
        return legacy_cache


    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        if self.key_cache is not None:
            return len(self.key_cache)
        else:
            return 0

    @property
    def seen_tokens(self):
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None



@dataclass
class SepLLMTrainingFreePress(BasePress):
    """
    This corresponds to the most fundamental usage of the training-free KV compression method described in the [SepLLM paper - ICML 2025](https://arxiv.org/abs/2412.12094). 
    In the training phase, SepLLM condenses the segment information into the KV of the separator that divides the segment. In the inference phase, the 
    corresponding `SepCache` only needs to store the KVs of initial tokens, separator tokens, and recent tokens for generation.

    For more advanced usage, such as setting parameters on a layer-wise basis, please refer to the original paper or the GitHub repository https://github.com/HKUDS/SepLLM.
    
    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression, which controls the size of the whole cache.
    init_cache_size_ratio : float, default=0.01
        The percentage of initial tokens to always preserve (sink tokens) in the whole cache (i.e., `cache_size`).
    sep_cache_size_ratio : float, default=0.2
        The ratio of `sep_cache_size` to the whole `cache_size`, which controls the size of the separator cache size.
    local_size_ratio : float, default=0.78
        The ratio of `local_size_ratio` to the whole `cache_size`, which controls the size of local window.

    
    """
    ## These settings are just for reference, not optimal.
    compression_ratio: float = 0.0
    init_cache_size_ratio: float = 0.0025
    sep_cache_size_ratio: float = 0.01
    local_size_ratio: float = 0.985
    
    separator_token_ids: Tuple[int] = (128000, 13, 11, 30, 0, 26, 25, 198, 220, 662, 1174, 949, 758, 2652, 551, 720, 256,262) ## Required for initialization of `SepCache` if `model_type` is not provided. The default value is an example for the tokenizer of llama3 8b series. Please provide the correct `separator_token_ids`` specific to the tokenizer of the current model.
    """
    The token ids of the separator tokens for the current model's tokenizer.            
    We have some examples, such as the Llama-3 series models, where setting `model_type='llama'` allows you 
        to skip setting `separator_token_ids` and `PADDING_ID` (`SepCache` will auto-fill them).
    """

    PADDING_ID: int = 128009 ## Required for initialization of `SepCache` if `model_type` is not provided. The default value is an example for the tokenizer of llama3 8b series. Please provide the correct `PADDING_ID`` specific to the tokenizer of the current model, or just set it to the id of "<end_of_sentence>".        
    """
    The token id of the padding token. You can just set `PADDING_ID` to the id of "<|endoftext|>" token of the tokenizer for the pretrained model.
    """

    model_type : str = 'llama' ## The model type for running the example. Choose from ['llama', 'pythia','falcon'].

    SEP_ACCUMULATION: bool = True
    USE_MAX_SEP_CACHE: bool = False

    _min_sink_cache_size : int = 4 # for attention sink
    _past_window_cache_offset : int = 16 # leave extra space for the past window buffer, which is to be compressed.


    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default forward hook called after the forward pass of an attention layer.

        This hook automatically applies compression during the pre-filling phase by:
        1. Checking if we're still in pre-filling (not generation) phase
        2. Extracting keys and values from the cache (handling quantization)
        3. Calling the compress method to reduce the cache size
        4. Updating the cache with compressed keys and values

        The hook ensures compression is only applied during pre-filling and correctly
        handles both quantized and unquantized caches.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer.
        input : list[torch.Tensor]
            Input tensors to the forward pass of the attention layer. This parameter
            is provided by PyTorch's hook mechanism but not used in the default implementation.
        kwargs : dict
            Keyword arguments passed to the attention layer's forward method, including:
            - hidden_states: Input embeddings to the attention layer
            - past_key_value: The KV cache object being modified
            - cache_position: Position indices indicating where we are in the sequence
            - position_embeddings: RoPE embeddings if applicable
        output : list
            Output from the attention layer's forward pass. Contains:
            - [0]: Hidden states output
            - [1]: Attention weights (may be None)

        Returns
        -------
        list
            The potentially modified output from the forward pass. This
            is the same as the input output, but the underlying cache has been compressed in-place.
        """

        if "input_ids_4_sepllm" in kwargs:
            input_ids = kwargs['input_ids_4_sepllm'] # For Training-Free SepLLM
        elif "flash_attn_kwargs" in kwargs:
            assert "input_ids_4_sepllm" in kwargs['flash_attn_kwargs'], f"The dict `kwargs['flash_attn_kwargs']` must have the key `input_ids_4_sepllm` to pass `input_ids`."
            input_ids = kwargs['flash_attn_kwargs']["input_ids_4_sepllm"] # For Training-Free SepLLM
        else:
            assert ('input_ids_4_sepllm' in kwargs) or ("flash_attn_kwargs" in kwargs), f"`input_ids` must be provided through key 'input_ids_4_sepllm' in `kwargs`. To achieve this, you may need to modify `modeling_xxx.py`.\
            E.g., in `modeling_llama.py`, you can just add `flash_attn_kwargs['input_ids'] = input_ids` in the `forward` function of `LlamaModel` in order to \
                eventually pass `input_ids` to the `forward` function of `LlamaAttention` through `flash_attn_kwargs`. Alternatively, you can manually modify the \
                `modeling_xxx.py` file, passing the `input_ids` parameter down the path from `XXXForCausalLM`, `XXXModel`, `XXXDecoderLayer` all the way to `XXXAttention` in all `forward` functions. \
            The above modifications are based on transformers>=4.53.0.\
            "

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_value"]
        q_len = hidden_states.shape[1]

        # Don't compress after pre-filling
        if kwargs["cache_position"][-1] > q_len:
            return output

        if isinstance(cache, QuantizedCache):
            keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
            values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])
        else:
            keys = cache.key_cache[module.layer_idx]
            values = cache.value_cache[module.layer_idx]

        keys, values = self.compress(module, hidden_states, keys, values, output[1], input_ids, kwargs)

        if isinstance(cache, QuantizedCache):
            cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=cache.axis_key)
            cache._quantized_value_cache[module.layer_idx] = cache._quantize(values, axis=cache.axis_value)
            cache.key_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache.value_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache._seen_tokens = keys.shape[2]
        else:
            cache.key_cache[module.layer_idx] = keys
            cache.value_cache[module.layer_idx] = values

        return output


    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        input_ids: torch.Tensor,
        separator_token_ids: List[int] = None , ## Required for initialization of `SepCache` if `model_type` is not provided. The default value is an example for the tokenizer of llama3 8b series. Please provide the correct `separator_token_ids`` specific to the tokenizer of the current model.
        PADDING_ID: int = None, ## Required for initialization of `SepCache` if `model_type` is not provided. The default value is an example for the tokenizer of llama3 8b series. Please provide the correct `PADDING_ID`` specific to the tokenizer of the current model, or just set it to the id of "<end_of_sentence>".        
        model_type: str = None, ## The model type for running the example. Choose from ['llama', 'pythia','falcon'].
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This corresponds to the most fundamental usage of the training-free KV compression method described in the [SepLLM paper - ICML 2025](https://arxiv.org/abs/2412.12094). 
        In the training phase, SepLLM condenses the segment information into the KV of the separator that divides the segment. In the inference phase, the 
        corresponding SepCache only needs to store the KVs of initial tokens, separator tokens, and recent tokens for generation.

        This `compress` function uses the `SepCache` to do the compression. Although `SepCache` can be used independently as a caching class during the inference stage of transformers models, 
            here, to meet the interface requirements of the KVPress repo, we did not directly use `SepCache` as the model's `past_key_value`. Instead, we simply borrowed the `update` function 
            of `SepCache` temporarily to achieve KV compression. Therefore, in practice, the KV pairs are not actually stored by the temporary SepCache object `tmp_sep_cache. Hence, when calling 
            the `update` function, setting `layer_idx = 0` is sufficient, without needing to differentiate which layer the current KV needs to be cached in the `SepCache`.        
        """          
        if separator_token_ids is None:  
            separator_token_ids = list(self.separator_token_ids)        
        if PADDING_ID is None:
            PADDING_ID = self.PADDING_ID 
        if model_type is None:
            model_type = self.model_type

        q_len = hidden_states.shape[1]
        
        layer_idx = 0 # We just use the 0-th cache layer's function to compress the KV cache only.

        cache_size = int( q_len * (1 - self.compression_ratio))
        local_size = int(cache_size * self.local_size_ratio)
        init_cache_size = max( int(cache_size * self.init_cache_size_ratio), self._min_sink_cache_size)
        sep_cache_size = max(int(cache_size * self.sep_cache_size_ratio), self._past_window_cache_offset - self._min_sink_cache_size)      

        assert self._past_window_cache_offset > 0, f"`_past_window_cache_offset` > 0, i.e., leave extra space for the past window buffer, which is to be compressed."
        assert self._past_window_cache_offset > self._min_sink_cache_size, f"`_past_window_cache_offset` should be greater than `_min_sink_cache_size`."
        assert cache_size > 0, f"The size of the whole preserved KV cache should be > 0."
        assert init_cache_size > 0, f"The size of initial KV cache should be > 0."
        assert sep_cache_size > 0, f"The size of separator KV cache should be > 0."

        tmp_sep_cache = SepCache(init_cache_size=init_cache_size, 
                                 sep_cache_size=sep_cache_size, 
                                 local_size=local_size, 
                                 cache_size=cache_size + self._past_window_cache_offset,  # leave extra space for the past window buffer, which is to be compressed.
                                 layer_num=32,  ## Any positive integer should be ok since we only use the 0-th cache layer of `SepCache` to compress the KV, not really using `SepCache` to store KV.
                                 USE_MAX_SEP_CACHE=self.USE_MAX_SEP_CACHE, 
                                 SEP_ACCUMULATION = self.SEP_ACCUMULATION,
                                 APPLY_PES_INSIDE=False, 
                                 separator_token_ids = separator_token_ids,
                                 PADDING_ID = PADDING_ID,                                 
                                 model_type=model_type)  
                    
        keys_prefill, keys_generate = keys[..., :-1 , :], keys[..., -1: , :]
        values_prefill, values_generate = values[..., :-1 , :], values[..., -1: , :]

        tmp_sep_cache.update(                
                    key_states = keys_prefill,
                    value_states = values_prefill,    
                    input_ids = input_ids[..., :-1],
                    layer_idx = layer_idx,     
                    PREFILLING_FLAG = True
                    )

        keys, values = tmp_sep_cache.update(                
                    key_states = keys_generate,
                    value_states = values_generate,    
                    input_ids = input_ids[..., -1:],
                    layer_idx = layer_idx,     
                    PREFILLING_FLAG = False
                    )        

        return keys, values

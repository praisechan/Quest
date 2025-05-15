from datasets import load_dataset
from torch.nn import CrossEntropyLoss

device = "cuda"

from MagicDec.Engine.SnapKV.model import Transformer
from MagicDec.Engine.utils import load_model_snapKV
import flashinfer

import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
from evaluation.quest_attention import enable_quest_attention_eval
from evaluation.llama import enable_tuple_kv_cache_for_llama 
from evaluation.mistral import enable_tuple_kv_cache_for_mistral

import copy

class LMBackend_Quest:
    def __init__(self, dtype = torch.bfloat16, device: str = "cuda:0", dec_len: int = 1, draft_dec_len: int = None) -> None:
        self.dtype = dtype
        self.device = device
        self.dec_len = dec_len
        self.model_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: model(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        self.prefill = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last=None, draft_paged_kv_indptr=None, draft_paged_kv_indices=None, draft_paged_kv_last_page_len=None: model.prefill(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, is_last, draft_paged_kv_indptr, draft_paged_kv_indices, draft_paged_kv_last_page_len)
        self.cachelens = None
        self.is_spec = False
        if draft_dec_len != None:
            self.is_spec = True
            self.draft_cachelens = None
            self.model_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, draft_kv_page_indices, draft_kv_page_indptr, draft_kv_page_lastlen: model.verify(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, draft_kv_page_indices, draft_kv_page_indptr, draft_kv_page_lastlen)
            self.draft_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: model.draft_forward(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        
        # for Quest
        self.draft_past_key_values = None
        self.input_tokens = None
        self.verified_cachelength = 0

    # def load_model(self, checkpoints: str, use_tp: bool, rank_group=None, group = None):
    #     self.model: Transformer = load_model_snapKV(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp=use_tp, rank_group=rank_group, group=group)        

    def load_model(self, model_path: str):
        if 'llama' in model_path.lower() or 'longchat' in model_path.lower():
            enable_tuple_kv_cache_for_llama()
        if 'mistral' in model_path.lower():
            enable_tuple_kv_cache_for_mistral()
      
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model = model.eval()


    def load_draft_model(self, model_path: str, draft_budget, chunk_size, bsz, max_len, latest_k=0):
        self.input_tokens = torch.zeros(bsz, max_len+1, device="cuda").long()
        self.cachelens = torch.zeros(bsz, dtype=torch.int32, device=self.device)
        args = argparse.Namespace(
            quest=True,
            token_budget=draft_budget,
            chunk_size=chunk_size,
            latest_k=latest_k
        )
                
        def load_model_and_tokenizer(model_path):
            if 'llama' in model_path.lower() or 'longchat' in model_path.lower():
                enable_tuple_kv_cache_for_llama()
            if 'mistral' in model_path.lower():
                enable_tuple_kv_cache_for_mistral()

            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
            )
            model = model.eval()

            enable_quest_attention_eval(model, args)

            return model

        if not hasattr(self, "model"):
            self.load_model(model_path)                

        self.draft_model = copy.copy(self.model)
        enable_quest_attention_eval(self.draft_model, args)

        # self.draft_model = load_model_and_tokenizer(model_path)

    # Only used for target verification
    @torch.inference_mode()
    def verify(self, input_ids: torch.LongTensor, benchmark = False):
            dec_len = input_ids.shape[1]
            # self.pre_verify(dec_len=dec_len)
            outputs = self.model(
                input_ids,
                past_key_values=self.draft_past_key_values,
                use_cache=True,
                output_all_token=True
            )
            
            self.cachelens += dec_len
            if benchmark:
                # If benchmarking the latency, don't update the cachelens and page table
                self.cachelens -= dec_len
                self.paged_kv_last_page_len -= dec_len
            return torch.argmax(outputs.logits, dim=-1)

    @torch.inference_mode()
    def speculate(self, input_ids: torch.LongTensor, bsz, gamma, benchmark = False):
      tokens_buffer= torch.zeros((bsz, gamma), device="cuda").long()
      draft_past_key_values = self.draft_past_key_values
      next_input_token = input_ids
      for i in range(gamma):
        outputs = self.draft_model(
            next_input_token,
            past_key_values=draft_past_key_values,
            use_cache=True,
        )
        
        next_input_token = torch.argmax(outputs.logits, dim=-1)
        tokens_buffer[:, i:i+1] = next_input_token
        draft_past_key_values = outputs.past_key_values
      
      return tokens_buffer
    
    @torch.inference_mode()
    def draft_kv_update(self, input_ids: torch.LongTensor):
        input_from_prefill = torch.concat((self.input_tokens[:, :self.verified_cachelength], input_ids), dim=1)
        outputs = self.draft_model(
            input_from_prefill,
            past_key_values=None,
            use_cache=True,
        )
        self.verified_cachelength += input_ids.shape[1]
        self.input_tokens[:,:self.verified_cachelength] = input_from_prefill
        self.draft_past_key_values = outputs.past_key_values

    @torch.inference_mode()
    def encode(self, input_ids: torch.LongTensor, benchmark = False):        
        # for Quest
        outputs = self.model(
            input_ids,
            past_key_values=None,
            use_cache=True,
        )
        self.draft_past_key_values = outputs.past_key_values
        self.use_verified_kv = True
        self.input_tokens[:,:input_ids.shape[1]] = input_ids
        self.verified_cachelength = input_ids.shape[1]
        self.cachelens = input_ids.shape[1]
        
        return torch.argmax(outputs.logits, dim=-1)
    
        
    # def compile(self):
    #     import torch._dynamo.config
    #     import torch._inductor.config
    #     torch._inductor.config.coordinate_descent_tuning = True
    #     torch._inductor.config.triton.unique_kernel_names = True
    #     torch._inductor.config.fx_graph_cache = True
    #     torch._functorch.config.enable_autograd_cache = True
    #     self.model_forward = torch.compile(self.model_forward, mode="max-autotune", fullgraph=True)
    #     if self.is_spec:
    #         self.draft_forward = torch.compile(self.draft_forward, mode="max-autotune", fullgraph=True)

    # # Only used for baseline inference
    # @torch.inference_mode()
    # def inference(self, input_ids: torch.LongTensor, benchmark = False):
    #         dec_len = input_ids.shape[1]
    #         self.pre_decode(dec_len=dec_len)

    #         logits = self.model_forward(
    #             model=self.model, 
    #             x=input_ids,
    #             input_pos=self.cachelens, 
    #             kv_append_indptr = self.qo_indptr*dec_len, kv_page_indices = self.paged_kv_indices, kv_page_indptr= self.paged_kv_indptr, kv_page_lastlen = self.paged_kv_last_page_len)
            
    #         self.cachelens += dec_len
    #         if benchmark:
    #             # If benchmarking the latency, don't update the cachelens and page table
    #             self.cachelens -= dec_len
    #             self.paged_kv_last_page_len -= dec_len
    #         return logits
    
    # def pre_decode(self, dec_len):
    #         self.paged_kv_last_page_len += dec_len
    #         self.decode_wrapper.plan(
    #             qo_indptr=self.qo_indptr*dec_len,
    #             paged_kv_indptr=self.paged_kv_indptr,
    #             paged_kv_indices=self.paged_kv_indices,
    #             paged_kv_last_page_len=self.paged_kv_last_page_len,
    #             num_qo_heads=self.model.config.n_head, 
    #             num_kv_heads=self.model.config.n_local_heads, 
    #             head_dim=self.model.config.head_dim, 
    #             page_size=self.page_size, 
    #             q_data_type=self.dtype, 
    #             causal=True,
    #         )
    
    # def pre_verify(self, dec_len):
    #         self.paged_kv_last_page_len += dec_len
    #         self.draft_paged_kv_last_page_len += 1
    #         self.draft_cachelens += 1

    #         self.decode_wrapper.plan(
    #             qo_indptr=self.qo_indptr*dec_len,
    #             paged_kv_indptr=self.paged_kv_indptr,
    #             paged_kv_indices=self.paged_kv_indices,
    #             paged_kv_last_page_len=self.paged_kv_last_page_len,
    #             num_qo_heads=self.model.config.n_head, 
    #             num_kv_heads=self.model.config.n_local_heads, 
    #             head_dim=self.model.config.head_dim, 
    #             page_size=self.page_size, 
    #             q_data_type=self.dtype, 
    #             causal=True,
    #         )
    
    # def pre_spec(self, dec_len):
    #         self.draft_paged_kv_last_page_len += dec_len
    #         self.draft_wrapper.plan(
    #             qo_indptr=self.qo_indptr*dec_len,
    #             paged_kv_indptr=self.draft_paged_kv_indptr,
    #             paged_kv_indices=self.draft_paged_kv_indices,
    #             paged_kv_last_page_len=self.draft_paged_kv_last_page_len,
    #             num_qo_heads=self.model.config.n_head, 
    #             num_kv_heads=self.model.config.n_local_heads, 
    #             head_dim=self.model.config.head_dim, 
    #             page_size=self.page_size, 
    #             q_data_type=self.dtype, 
    #             causal=True,
    #         )
        
    
    # def pre_encode(self, dec_len):
    #     self.num_pages_per_request+=1
    #     qo_indptr = self.qo_indptr*dec_len
    #     self.paged_kv_indices = torch.cat([torch.arange(i * self.max_num_pages_per_request, i * self.max_num_pages_per_request + self.num_pages_per_request[i], dtype=torch.int32, device=self.device) for i in range(self.batch_size)])
    #     self.paged_kv_indptr[1:] = torch.cumsum(self.num_pages_per_request, dim=0, dtype=torch.int32)
    #     self.paged_kv_last_page_len = torch.full((self.batch_size,), dec_len, dtype=torch.int32, device=self.device)
    #     self.prefill_wrapper.plan(
    #         qo_indptr=qo_indptr,
    #         paged_kv_indptr=self.paged_kv_indptr,
    #         paged_kv_indices=self.paged_kv_indices,
    #         paged_kv_last_page_len=self.paged_kv_last_page_len,
    #         num_qo_heads=self.model.config.n_head, 
    #         num_kv_heads=self.model.config.n_local_heads, 
    #         head_dim=self.model.config.head_dim, 
    #         page_size=self.page_size, 
    #         q_data_type=self.dtype, 
    #         causal=True
    #         )
          
    
    # @torch.inference_mode()
    # def clear_kv(self):
    #     for b in self.model.layers:
    #         b.attention.kv_cache.kv_cache.zero_()
    #         if self.is_spec:
    #             b.attention.kv_cache.draft_cache.zero_()
    #     self.cachelens.zero_()
    #     self.qo_indptr = torch.arange(self.batch_size+1, dtype=torch.int32, device=self.device)
    #     self.paged_kv_indptr = torch.arange(self.batch_size+1, dtype=torch.int32, device=self.device)
    #     self.paged_kv_indices = torch.empty(self.max_num_pages, dtype=torch.int32, device=self.device)
    #     self.paged_kv_last_page_len = torch.zeros((self.batch_size), dtype=torch.int32, device=self.device)
    #     self.num_pages_per_request = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
    #     if self.is_spec:
    #         self.draft_cachelens.zero_()
    #         self.draft_paged_kv_indptr = torch.arange(self.batch_size+1, dtype=torch.int32, device=self.device)*(self.draft_budget//self.page_size + 1)
    #         self.draft_paged_kv_indices = torch.arange(self.draft_num_pages, dtype=torch.int32, device=self.device)
    #         self.draft_paged_kv_last_page_len = torch.ones((self.batch_size), dtype=torch.int32, device=self.device)

    
    # @torch.inference_mode()
    # def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048, draft_budget = 0, window_size = 32):
    #     self.max_length = max_seq_length
    #     self.batch_size = max_batch_size
    #     self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
    #     # Prefill length should be devisible by 128 and plus 1 or window_size
    #     # Max Length should be divisible by 128
    #     page_size = 128
    #     max_num_pages = max_batch_size * max_seq_length // page_size
    #     if max_num_pages*page_size < max_batch_size*max_seq_length:
    #         max_num_pages += max_batch_size
    #     self.max_num_pages_per_request = max_num_pages // max_batch_size
    #     self.num_pages_per_request = torch.zeros(max_batch_size, device=self.device, dtype=torch.int32)
    #     self.page_size = 128
    #     self.max_num_pages = max_num_pages


    #     # Init Target Attention Backend(Flashinfer)
    #     self.decode_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
    #     self.prefill_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)

    #     self.qo_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
    #     self.paged_kv_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)
    #     self.paged_kv_indices = torch.empty(max_num_pages, dtype=torch.int32, device=self.device)
    #     self.paged_kv_last_page_len = torch.zeros((max_batch_size), dtype=torch.int32, device=self.device)
    #     self.decode_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.decode_buffer, "NHD", use_cuda_graph=True,
    #                                                                           qo_indptr_buf=self.qo_indptr, 
    #                                                                           paged_kv_indptr_buf=self.paged_kv_indptr, 
    #                                                                           paged_kv_indices_buf=self.paged_kv_indices, 
    #                                                                           paged_kv_last_page_len_buf=self.paged_kv_last_page_len)
        
    #     self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.prefill_buffer, "NHD")
    #     torch.library.define(
    #         "mylib::target_decode",
    #         "(Tensor q, Tensor kv_cache) -> Tensor",
    #     )
    #     @torch.library.impl("mylib::target_decode", "cuda")
    #     def target_decode(q, kv_cache):
    #         return self.decode_wrapper.run(
    #             q, kv_cache
    #         )
    #     @torch.library.register_fake("mylib::target_decode")
    #     def target_decode_abstract(q, kv_cache):
    #         return torch.empty_like(q)
        
    #     torch.library.define(
    #         "mylib::target_prefill",
    #         "(Tensor q, Tensor kv_cache) -> Tensor",
    #     )
    #     @torch.library.impl("mylib::target_prefill", "cuda")
    #     def target_prefill(q, kv_cache):
    #         return self.prefill_wrapper.run(
    #             q, kv_cache
    #         )
    #     @torch.library.register_fake("mylib::target_prefill")
    #     def target_prefill_abstract(q, kv_cache):
    #         return torch.empty_like(q)

    #     # If using speculative decoding, init draft attention backend
    #     if self.is_spec:
    #         self.draft_budget = draft_budget
    #         self.draft_cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
    #         self.draft_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
    #         self.draft_num_pages = (draft_budget//page_size + 1)*max_batch_size
    #         self.draft_paged_kv_indptr = torch.arange(max_batch_size+1, dtype=torch.int32, device=self.device)*(draft_budget//page_size + 1)
    #         self.draft_paged_kv_indices = torch.arange(self.draft_num_pages, dtype=torch.int32, device=self.device)
    #         self.draft_paged_kv_last_page_len = torch.ones((max_batch_size), dtype=torch.int32, device=self.device)
    #         self.draft_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.draft_buffer, "NHD", use_cuda_graph=True,
    #                                                                             qo_indptr_buf=self.qo_indptr, 
    #                                                                             paged_kv_indptr_buf=self.draft_paged_kv_indptr, 
    #                                                                             paged_kv_indices_buf=self.draft_paged_kv_indices, 
    #                                                                             paged_kv_last_page_len_buf=self.draft_paged_kv_last_page_len)
    #         torch.library.define(
    #             "mylib::draft_decode",
    #             "(Tensor q, Tensor kv_cache) -> Tensor",
    #         )
    #         @torch.library.impl("mylib::draft_decode", "cuda")
    #         def draft_decode(q, kv_cache):
    #             return self.draft_wrapper.run(
    #                 q, kv_cache
    #             )
    #         @torch.library.register_fake("mylib::draft_decode")
    #         def draft_decode_abstract(q, kv_cache):
    #             return torch.empty_like(q)

    #     if self.is_spec:
    #         with torch.device(self.device):
    #             self.model.setup_caches(num_pages=max_num_pages, page_size=page_size, spec=self.is_spec, draft_num_pages = self.draft_num_pages, draft_budget = draft_budget, window_size = window_size)
    #     else:
    #         with torch.device(self.device):
    #             self.model.setup_caches(num_pages=max_num_pages, page_size=page_size)
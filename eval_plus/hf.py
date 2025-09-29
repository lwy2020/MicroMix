from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from base import DecoderBase
from utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)

        
class HuggingFaceDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        dataset: str,
        force_base_prompt: bool = False,
        attn_implementation: str = "eager",
        device_map: str = None,
        gguf_file: str = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    
        # from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        # 定义FP8量化配置
        # fp8_config = BitsAndBytesConfig(
        #     load_in_8bit=True,  # 启用8位量化
        #     bnb_8bit_compute_dtype=torch.float8_e4m3fn  # 使用FP8计算类型
        # )
        
        # kwargs = {            
        #     "device_map": "auto",  # 自动分配设备（推荐，替代"cpu"以利用GPU）
        #     "trust_remote_code": self.trust_remote_code,            
        #     "torch_dtype": getattr(torch, self.dtype),  # 可保留为基础数据类型
        #     "attn_implementation": attn_implementation,  # 保持原注意力实现
        #     "gguf_file": gguf_file,
        #     # 添加FP8量化配置
        #     "quantization_config": fp8_config,
        #     # 可选：启用模型并行以优化大模型加载
        #     "device_map": "auto" if torch.cuda.device_count() > 1 else "cuda:0"
        # }        
        # kwargs = {
        #     "device_map": "cpu",
        #     "trust_remote_code": self.trust_remote_code,
        #     "torch_dtype": getattr(torch, self.dtype),
        #     "attn_implementation": attn_implementation,  # "eager", "flash_attention_2", "sdpa"
        #     "gguf_file": gguf_file
        # }
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.skip_special_tokens = True

        print(f"{kwargs = }")

        self.force_base_prompt = force_base_prompt

        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        tokenizer_kwargs = {}
        if gguf_file is not None:
            tokenizer_kwargs["gguf_file"] = gguf_file
        self.tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)
        if self.is_direct_completion():  # no chat template
            self.eos += extra_eos_for_direct_completion(dataset)
        else:  # with chat template
            self.eos += ["\n```\n"]
        model_name = name.split('/')[-2]
        print(f"{self.eos = }")
        
    
        
        # from modelutils_qwen import reorder_model_qwen
        
        # index_filename = f'/root/MicroMix/saved/{model_name}_reorder_index_wikitext2_mean.pt'
        # p6_num_filename = f'/root/MicroMix/saved/{model_name}_p6_num_wikitext2_mean.pt'
        # p8_num_filename = f'/root/MicroMix/saved/{model_name}_p8_num_wikitext2_mean.pt'
        # reorder_index = torch.load(index_filename, weights_only=False)
        # p6_nums = torch.load(p6_num_filename, weights_only=False)
        # p8_nums = torch.load(p8_num_filename, weights_only=False)
        
        # self.model = reorder_model_qwen(self.model, device='cuda', kv_cache=False, reorder_index=reorder_index, p8_nums=p8_nums, p6_nums=p6_nums)
        
        # self.model = self.model.to(self.device)
        print(self.model)
    def is_direct_completion(self) -> bool:
        return self.force_base_prompt or self.tokenizer.chat_template is None

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
            )
        )
        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature

        outputs = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            stop_strings=self.eos,
            tokenizer=self.tokenizer,
            **kwargs,
        )

        gen_strs = self.tokenizer.batch_decode(
            outputs[:, input_tokens.size(-1) :],
            skip_special_tokens=self.skip_special_tokens,
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index].replace("\t", "    "))
        return outputs
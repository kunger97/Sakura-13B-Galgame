import logging
from pathlib import Path
from pprint import pprint

from neural_speed import Model
from transformers import GenerationConfig

import utils
from infers import BaseInferEngine
from utils.model import SakuraModelConfig

logger = logging.getLogger(__name__)

class NeuralSpeed(BaseInferEngine):
    def __init__(self, args: SakuraModelConfig):
        from transformers import AutoConfig
        logger.warn("NeuralSpeed is being used for generation. Due to missing parameters, the generation results may be incorrect.")
        NEModel = Model();
        #Choose the best compute_dtype and scal_dtype see this website: https://github.com/intel/neural-speed/blob/main/neural_speed/core/README.md
        NEModel.init(args.model_name_or_path, alg="asym", group_size=128, scale_dtype="fp32", compute_dtype="int8")
        self.model = NEModel
        return

    def get_metadata(self, args: SakuraModelConfig):
        file = Path(args.model_name_or_path)
        filename = file.stem
        metadata = filename.split("-")
        model_version, model_quant = metadata[-2], metadata[-1]
        model_name = "-".join(metadata[:-2])
        return model_name, model_version, model_quant

    def generate(self, tokenizer, prompt: str, model_version: str,
                        generation_config: GenerationConfig):
        input_tokens = tokenizer(prompt, return_tensors="pt")

        input_tokens_len = input_tokens.input_ids.shape[-1]

        import torch
        from typing import List
        from transformers import StoppingCriteria, StoppingCriteriaList

        # Todo - 判断EOS Token来停止模型输出。
        class StopOnTokens(StoppingCriteria):
            def __init__(self, start_length, stop_token_id: List[int], max_new_tokens):
                self.start_length = start_length
                self.stop_token_id = stop_token_id
                self.max_new_tokens = max_new_tokens

            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
            ) -> bool:
                #print(f"起始Token{self.start_length}，当前Token数量{input_ids.shape[-1] - self.start_length}")
                if input_ids.shape[-1] - self.start_length > self.max_new_tokens:
                    return True
                for stop_id in self.stop_token_id:
                        if input_ids[0][input_ids.shape[-1] - 1] == stop_id:
                            return True
                return False
            
        if model_version == "0.9":
            stopping_criteria = StoppingCriteriaList(
                [
                    StopOnTokens(
                        start_length=input_tokens.input_ids.shape[1],
                        stop_token_id=[151645], #这里写死了QWEN的模型EOS Token
                        max_new_tokens=generation_config.__dict__['max_new_tokens'],
                    )
                ]
            )
        else:
            stopping_criteria = StoppingCriteriaList(
                [
                    StopOnTokens(
                        start_length=input_tokens.input_ids.shape[1],
                        stop_token_id=[tokenizer.eos_token_id],
                        max_new_tokens=generation_config.__dict__['max_new_tokens'],
                    )
                ]
            )

        output = self.model.generate(input_tokens.input_ids, stopping_criteria=stopping_criteria, ctx_size=8192, max_new_tokens=generation_config.__dict__['max_new_tokens'], repetition_penalty=1.2, temperature=0.85, top_p=0.9, do_sample=True)[0]
        response = tokenizer.decode(output)
        response = utils.split_response(response, model_version)

        new_tokens = len(output) - input_tokens_len
        return response, (input_tokens_len, new_tokens)

    def stream_generate(self, tokenizer, prompt: str, model_version: str,
                        generation_config: GenerationConfig):
        from transformers import TextStreamer

        logger.debug(f"prompt is: {prompt}")
        input_tokens = tokenizer(prompt, return_tensors="pt")
        streamer = TextStreamer(tokenizer)

        import torch
        from typing import List
        from transformers import StoppingCriteria, StoppingCriteriaList

        # Todo - 判断EOS Token来停止模型输出。
        class StopOnTokens(StoppingCriteria):
            def __init__(self, start_length, stop_token_id: List[int], max_new_tokens):
                self.start_length = start_length
                self.stop_token_id = stop_token_id
                self.max_new_tokens = max_new_tokens

            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
            ) -> bool:
                #print(f"起始Token{self.start_length}，当前Token数量{input_ids.shape[-1] - self.start_length}")
                if input_ids.shape[-1] - self.start_length > self.max_new_tokens:
                    return True
                for stop_id in self.stop_token_id:
                        if input_ids[0][input_ids.shape[-1] - 1] == stop_id:
                            return True
                return False
            
        if model_version == "0.9":
            stopping_criteria = StoppingCriteriaList(
                [
                    StopOnTokens(
                        start_length=input_tokens.input_ids.shape[1],
                        stop_token_id=[151645], #这里写死了QWEN的模型EOS Token
                        max_new_tokens=generation_config.__dict__['max_new_tokens'],
                    )
                ]
            )
        else:
            stopping_criteria = StoppingCriteriaList(
                [
                    StopOnTokens(
                        start_length=input_tokens.input_ids.shape[1],
                        stop_token_id=[tokenizer.eos_token_id],
                        max_new_tokens=generation_config.__dict__['max_new_tokens'],
                    )
                ]
            )

        output = self.model.generate(input_tokens.input_ids, streamer=streamer, stopping_criteria=stopping_criteria, ctx_size=8192, max_new_tokens=generation_config.__dict__['max_new_tokens'], repetition_penalty=1.2, temperature=0.85, top_p=0.9, do_sample=True)[0]
        yield output, 'unknown'

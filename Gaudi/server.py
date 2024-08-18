from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import argparse
from utils import initialize_model
import torch
import logging
import time

app = Flask(__name__)

cors = CORS(app, resources={r"/v1/*": {
    "origins": ["*"],
    "methods": ["GET", "POST", "PUT", "DELETE"],
    "allow_headers": ["*"]
}})

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置模型路径变量
MODEL_PATH = "/home/u76dd8763cba39e0b015b7769dfb6510/SakuraLLM/Sakura_32B_V0.9"  # 请替换为实际的模型路径

# 初始化参数
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default=MODEL_PATH)
parser.add_argument("--bf16", action="store_true", default=True)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--use_kv_cache", action="store_true", default=True)
parser.add_argument("--attn_softmax_bf16", action="store_true", default=True)
parser.add_argument("--use_hpu_graphs", action="store_true", default=True)
parser.add_argument("--use_flash_attention",action="store_true", default=True)
parser.add_argument("--flash_attention_recompute",action="store_true")
parser.add_argument("--flash_attention_causal_mask", action="store_true", default=True)
parser.add_argument("--flash_attention_fast_softmax",action="store_true")
parser.add_argument("--ignore_eos", action="store_false", default=False)
parser.add_argument("--device", type=str, default="hpu")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=100)
parser.add_argument("--max_input_tokens", type=int, default=0)
parser.add_argument("--n_iterations", type=int, default=5)
parser.add_argument("--limit_hpu_graphs", action="store_true", default=False)
parser.add_argument("--torch_compile", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--token",default=None,type=str)
parser.add_argument("--world_size", type=int, default=0)
parser.add_argument("--quant_config", type=str, default="")
parser.add_argument("--assistant_model",default=None,type=str)
parser.add_argument("--model_revision",default="main",type=str)
parser.add_argument("--trust_remote_code",action="store_true")
parser.add_argument("--load_quantized_model",action="store_true")
parser.add_argument("--disk_offload", action="store_true")
parser.add_argument("--peft_model",default=None,type=str)
parser.add_argument("--bad_words",default=None,type=str,nargs="+")
parser.add_argument("--force_words",default=None,type=str,nargs="+")
parser.add_argument("--bucket_size",default=-1,type=int)
parser.add_argument("--bucket_internal",action="store_true")
parser.add_argument("--do_sample",action="store_true",default=True)
parser.add_argument("--num_beams",default=1,type=int )
parser.add_argument("--top_k",default=None,type=int,)
parser.add_argument("--penalty_alpha",default=None,type=float)
parser.add_argument("--trim_logits",action="store_true")
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument("--reuse_cache",action="store_true",default=True)
parser.add_argument("--reduce_recompile",action="store_true")
parser.add_argument("--const_serialization_path","--csp",type=str)
args = parser.parse_args()

print(args)

logger.info("Initializing model...")

try:
    # 初始化模型
    model, _, tokenizer, generation_config = initialize_model(args, logger)
    logger.info("Model initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    raise

# 存储对话历史的字典
conversation_history = {}

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    messages = data.get('messages', [])
    temperature = data.get('temperature', 1.0)
    top_p = data.get('top_p', 1.0)
    max_new_tokens = data.get('max_tokens', 16)
    frequency_penalty = data.get('frequency_penalty', 0.0)
    do_sample = data.get('do_sample', False)
    conversation_id = data.get('conversation_id', 'default')

    # 更新generation_config
    generation_config.temperature = temperature
    generation_config.top_p = top_p
    generation_config.max_new_tokens = max_new_tokens
    generation_config.repetition_penalty = 1.0 + frequency_penalty
    generation_config.do_sample = do_sample

    # 使用 apply_chat_template 方法
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 使用tokenizer.batch_encode_plus
    if args.max_input_tokens > 0:
        input_tokens = tokenizer.batch_encode_plus(
            [input_text],
            return_tensors="pt",
            padding="max_length",
            max_length=args.max_input_tokens,
            truncation=True,
        )
    else:
        input_tokens = tokenizer.batch_encode_plus([input_text], return_tensors="pt", padding=True)

    for t in input_tokens:
                    if torch.is_tensor(input_tokens[t]):
                        input_tokens[t] = input_tokens[t].to(args.device)
                        
    # 生成文本
    try:
        outputs = model.generate(
            **input_tokens,
            generation_config=generation_config,
            lazy_mode=True,
            hpu_graphs=args.use_hpu_graphs,
            ignore_eos=False
        ).cpu()
        
        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]        
        generated_text = generated_text.split("assistant\n")[-1]
        
        # 构造响应
        response = {
            "id": "chatcmpl-" + torch.randint(0, 1000000, (1,)).item().__str__(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": args.model_name_or_path,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text.strip()
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(input_tokens[0]),
                "completion_tokens": len(generated_text) - len(input_tokens[0]),
                "total_tokens": len(outputs[0])
            }
        }

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained("lmsys/longchat-7b-v1.5-32k")
model = AutoModel.from_pretrained("lmsys/longchat-7b-v1.5-32k")
# model = AutoModel.from_pretrained("/data/hf_cache/hub/models--lmsys--longchat-7b-v1.5-32k")

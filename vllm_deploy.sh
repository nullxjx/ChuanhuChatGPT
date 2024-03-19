## 注意，由于Tesla T4显卡只有16G显存，一次只能部署一个模型，所以下面命令只能选择一个模型部署
## 使用该脚本部署完模型之后，记得在config.json中更改default_model为

# 先手动在终端执行以下命令，激活虚拟环境
# conda activate vllm

# 部署Qwen模型，模型名字 Qwen1.5-4B-Chat
python -m vllm.entrypoints.openai.api_server --model /root/models/Qwen1.5-4B-Chat --dtype=float16 --max-model-len 8192 --served-model-name Qwen1.5-4B-Chat --port 6399

# 部署Yi模型，模型名字 Yi-6B-Chat
#python -m vllm.entrypoints.openai.api_server --model /root/models/Yi-6B-Chat --dtype=float16 --max-model-len 4096 --served-model-name Yi-6B-Chat --port 6399

# 部署chatglm，模型名字 chatglm3-6b-32k
#python -m vllm.entrypoints.openai.api_server --model /root/models/chatglm3-6b-32k --dtype=float16 --max-model-len 8192 --served-model-name chatglm3-6b-32k --port 6399 --trust-remote-code --gpu-memory-utilization 0.95
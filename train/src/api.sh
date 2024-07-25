CUDA_VISIBLE_DEVICES=0,1 API_PORT=8000 python api_demo.py\
    --model_name_or_path your_local_model_path\
    --template qwen/intern2/llama3\
    --infer_backend vllm\
    --vllm_enforce_eager
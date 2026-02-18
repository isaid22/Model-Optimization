#!/bin/bash

# Corrected polygraphy command
/home/ubuntu/ai_venv/bin/polygraphy run onnx_model/model.onnx \
  --onnxrt \
  --save-tactics onnxrt_tactics.json

# Corrected trtllm-build command for INT8 quantization
/home/ubuntu/ai_venv/bin/trtllm-build --onnx_file onnx_model/model.onnx \
            --output_dir ./engine_outputs_int8 \
            --log_level=error \
            --max_batch_size=1 \
            --max_input_len=128 \
            --max_output_len=128 \
            --builder_opt=0 \
            --gemm_plugin=float16 \
            --bert_attention_plugin=float16 \
            --gpt_attention_plugin=float16 \
            --remove_input_padding=enable \
            --paged_kv_cache=enable \
            --int8_mode

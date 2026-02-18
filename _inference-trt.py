import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# Load the TensorRT engine
logger = trt.Logger(trt.Logger.WARNING)
with open("mpnet.engine", 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# Prepare input
text = "This is a test sentence"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=384)
input_ids = inputs['input_ids'].numpy().astype(np.int32)
attention_mask = inputs['attention_mask'].numpy().astype(np.int32)

# Set input shapes
context.set_input_shape("input_ids", input_ids.shape)
context.set_input_shape("attention_mask", attention_mask.shape)

# Allocate GPU memory
d_input_ids = cuda.mem_alloc(input_ids.nbytes)
d_attention_mask = cuda.mem_alloc(attention_mask.nbytes)
d_output = cuda.mem_alloc(input_ids.shape[0] * 768 * 4)

# Copy inputs to GPU
cuda.memcpy_htod(d_input_ids, input_ids)
cuda.memcpy_htod(d_attention_mask, attention_mask)

# Run inference
context.execute_v2([int(d_input_ids), int(d_attention_mask), int(d_output)])

# Copy output back to CPU
output = np.empty([input_ids.shape[0], 768], dtype=np.float32)
cuda.memcpy_dtoh(output, d_output)

print("Output shape:", output.shape)
print("Output:", output)
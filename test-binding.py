import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
with open("mpnet.engine", 'rb') as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

print(f"Number of bindings: {engine.num_io_tensors}")
for i in range(engine.num_io_tensors):
    print(f"Binding {i}: {engine.get_tensor_name(i)}")
import numpy as np
import ray
from ray import serve
import tensorrt as trt
import pycuda.driver as cuda
from transformers import AutoTokenizer


class TRTInference:
    def __init__(self, engine_path):
        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        # Detach context from init thread; we will push/pop per call.
        self.ctx.pop()

    def infer(self, input_ids, attention_mask):
        self.ctx.push()
        try:
            self.context.set_input_shape("input_ids", input_ids.shape)
            self.context.set_input_shape("attention_mask", attention_mask.shape)

            output_mem = None
            output_shape = None
            d_output = None

            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                shape = self.context.get_tensor_shape(tensor_name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                size = trt.volume(shape)

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    if "input_ids" in tensor_name:
                        np.copyto(host_mem, input_ids.ravel())
                    elif "attention_mask" in tensor_name:
                        np.copyto(host_mem, attention_mask.ravel())
                    cuda.memcpy_htod_async(device_mem, host_mem, self.stream)
                else:
                    output_mem = host_mem
                    output_shape = shape
                    d_output = device_mem

                self.context.set_tensor_address(tensor_name, int(device_mem))

            self.context.execute_async_v3(self.stream.handle)

            cuda.memcpy_dtoh_async(output_mem, d_output, self.stream)
            self.stream.synchronize()

            output_mem.shape = tuple(output_shape)
            return output_mem
        finally:
            self.ctx.pop()

    def __del__(self):
        if hasattr(self, "ctx"):
            try:
                self.ctx.pop()
            except Exception:
                pass


@serve.deployment(ray_actor_options={"num_gpus": 1})
class MPNetTRTService:
    def __init__(self, engine_path="mpnet.engine"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )
        self.trt = TRTInference(engine_path)

    async def __call__(self, request):
        payload = await request.json()
        text = payload.get("text", "")
        max_length = int(payload.get("max_length", 128))

        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        input_ids = inputs["input_ids"].astype(np.int32)
        attention_mask = inputs["attention_mask"].astype(np.int32)

        embedding = self.trt.infer(input_ids, attention_mask)
        return {"embedding": embedding.tolist()}


def main():
    ray.init()
    serve.run(MPNetTRTService.bind())
    # Keep the process alive so the HTTP server stays up.
    import time
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()

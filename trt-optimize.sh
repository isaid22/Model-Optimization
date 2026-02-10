/usr/src/tensorrt/bin/trtexec \
  --onnx=onnx_model/model.onnx \
  --saveEngine=mpnet.engine \
  --fp16 \
  --minShapes=input_ids:1x1,attention_mask:1x1 \
  --optShapes=input_ids:1x128,attention_mask:1x128 \
  --maxShapes=input_ids:1x512,attention_mask:1x512
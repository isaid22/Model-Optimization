# Model-Optimization
Examples of model optimization work.

This work is done on AWS g5.4xl instance.

1. run `export-to-onnx.py`.
2. run `./trt-optimize.sh`
3. run `test-binding.py`
4. run `inference-trt.py`

# ONNX Conversion

ONNX Conversion of the native model is done by [`export-to-onnx.py`](./export-to-onnx.py). The conversion process will create a folder `onnx_model` in the current directory to save all the artifacts. Here are the contents within that folder:

Model files:

`model.onnx` — the exported ONNX graph
`config.json` — the model's architecture config

Tokenizer files:

`tokenizer.json` — the fast tokenizer
`tokenizer_config.json` — tokenizer settings
`vocab.txt` — the vocabulary
`special_tokens_map.json` — special token definitions


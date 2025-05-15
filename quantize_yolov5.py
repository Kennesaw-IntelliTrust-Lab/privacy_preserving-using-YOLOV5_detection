from onnxruntime.quantization import quantize_dynamic, QuantType

# Input and output ONNX model paths
model_fp32 = 'runs/train/exp13/weights/best.onnx'
model_quant = 'runs/train/exp13/weights/best_quantized.onnx'

# Apply dynamic quantization (weights to INT8)
quantize_dynamic(
    model_input=model_fp32,
    model_output=model_quant,
    weight_type=QuantType.QInt8  # Use QuantType.QUInt8 if needed
)

print(f"✅ Quantized model saved to {model_quant}")

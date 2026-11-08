# std/quant.t — Quantization & Dequantization Engine (FP16, Q4_0, Q8_0) cho T

let _sys = load("__sys")

fn quantize_f16(t):
    return _sys.quant_f16(t)

fn dequantize_f16(bytes_data, shape):
    return _sys.dequant_f16(bytes_data, shape)

fn quantize_bf16(t):
    return _sys.quant_bf16(t)

fn dequantize_bf16(bytes_data, shape):
    return _sys.dequant_bf16(bytes_data, shape)

fn quantize_f32(t):
    return _sys.quant_f32(t)

fn dequantize_f32(bytes_data, shape):
    return _sys.dequant_f32(bytes_data, shape)

fn quantize_q4_0(t):
    return _sys.quant_q4_0(t)

fn dequantize_q4_0(bytes_data, shape):
    return _sys.dequant_q4_0(bytes_data, shape)

fn quantize_q8_0(t):
    return _sys.quant_q8_0(t)

fn dequantize_q8_0(bytes_data, shape):
    return _sys.dequant_q8_0(bytes_data, shape)

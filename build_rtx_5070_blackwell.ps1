# ============================================================
#  LAMA TOUR - Build TensorRT-LLM pour RTX 5070 (Blackwell sm_100 / sn120)
#  NVFP4 pour les couches sures, FP16 pour les clusters d'outliers.
#  La liste FP16 est coherente avec precision_map.json.
#  Pre-requis : TensorRT-LLM installe + checkpoint genere dans ./trt_checkpoint
# ============================================================

$checkpointDir = "./trt_checkpoint"
$outputDir     = "./engine_blackwell_sn120"

# Couches ssm_conv1d verrouillees en FP16 (densite d'outliers >= 0.31)
$fp16Layers = @(
    "blk.0.ssm_conv1d.weight:fp16",
    "blk.1.ssm_conv1d.weight:fp16",
    "blk.2.ssm_conv1d.weight:fp16",
    "blk.4.ssm_conv1d.weight:fp16",
    "blk.8.ssm_conv1d.weight:fp16",
    "blk.18.ssm_conv1d.weight:fp16",
    "blk.25.ssm_conv1d.weight:fp16",
    "blk.26.ssm_conv1d.weight:fp16",
    "blk.28.ssm_conv1d.weight:fp16",
    "blk.29.ssm_conv1d.weight:fp16",
    "blk.30.ssm_conv1d.weight:fp16"
) -join ","

if (-not (Test-Path $checkpointDir)) {
    Write-Error "Checkpoint introuvable : $checkpointDir (generez-le d'abord avec TensorRT-LLM)"
    exit 1
}

trtllm-build --checkpoint_dir $checkpointDir --output_dir $outputDir --dtype float16 --quantization nvfp4 --use_gpt_attention_plugin enable --use_gemm_plugin enable --per_layer_precision $fp16Layers

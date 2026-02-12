import os

# --- Paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
INSWAPPER_PATH = os.path.join(MODELS_DIR, "inswapper_128.onnx")  #https://huggingface.co/Patil/inswapper/tree/main
INSIGHTFACE_ROOT = os.path.join(MODELS_DIR, "insightface")
GFPGAN_MODEL_PATH = os.path.join(MODELS_DIR, "GFPGANv1.4.pth")

# --- Capture & Processing ---
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
MAX_QUEUE_SIZE = 2

# --- Default UI Settings ---
DEFAULT_PROVIDER = "TensorRT" #"CUDA"#
DEFAULT_SHARPNESS = 0.4
DEFAULT_MASK_BLUR = 30
DEFAULT_OVAL_WIDTH = 0.45
DEFAULT_OVAL_HEIGHT = 0.55

# --- TensorRT Config ---
TENSORRT_CONFIG = {
    'device_id': 0,
    'trt_max_workspace_size': 4294967296,  # 4 GB
    'trt_fp16_enable': True,
    'trt_engine_cache_enable': True,
    'trt_engine_cache_path': os.path.join(MODELS_DIR, "trt_cache"),
    'trt_timing_cache_enable': True,
    'trt_timing_cache_path': os.path.join(MODELS_DIR, "trt_cache"),
    # ── Ключевые добавления для лучшей оптимизации ──
    'trt_builder_optimization_level': 3,          # 0..3, 3 — максимальная оптимизация (может дольше строить engine)
    'trt_timing_cache_enable': True,
    'trt_force_sequential_engine_build': False,
    'trt_dla_enable': False,
    

}

PROVIDERS = {
    "TensorRT": [
        ('TensorrtExecutionProvider', TENSORRT_CONFIG),
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
            'use_tf32': 1,
        })
    ]
}
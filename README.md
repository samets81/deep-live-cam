<h1 align="center">Deep Live Cam TensorRT - Real-time Face Swapping</h1>

<p align="center">
  High-performance real-time face swapping with TensorRT and GFPGAN
</p>

## 📋 Description

Deep Live Cam TensorRT is an optimized application for real-time face swapping from webcam. The project uses TensorRT for maximum performance on NVIDIA GPUs and includes face quality enhancement with GFPGAN.

### ✨ Key Features

- **Real-time face swapping** from webcam
- **TensorRT optimization** for maximum performance on NVIDIA GPUs
- **GFPGAN integration** for enhanced face quality
- **Multiple camera support** - choose from available webcams
- **Resolution settings** - from 320x240 to 1920x1080 (Full HD)
- **Flexible customization**:
  - Face sharpness (0.0 - 5.0)
  - Mask transition smoothness (1 - 99)
  - Oval mask size (width and height)
- **Real-time FPS display**
- **Mask caching** for improved performance
- **Multi-threaded processing** for optimal speed

## 🎯 Purpose

This tool is designed for:
- AI-generated media content creation
- Character animation
- Creative projects
- Educational purposes

## ⚠️ Disclaimer

**Important**: This software must be used responsibly and legally:

- **Ethical Use**: Obtain consent when using a real person's face
- **Content Labeling**: Clearly mark output as deepfake when sharing
- **Content Restrictions**: Built-in checks prevent processing inappropriate content
- **Legal Responsibility**: We are not responsible for end-user actions

By using this software, you agree to use it in a manner that respects the rights and dignity of others.

## 🚀 Quick Start

### System Requirements

- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GPU with CUDA support (developed for RTX 4060 and RTX 5070 sm_89, sm_120)
- **CUDA**: 12.x
- **cuDNN**: v9.18 for CUDA 12.x
- **Python**: 3.10 !!!
- **RAM**: minimum 8 GB (16 GB recommended)
- **Webcam**: any compatible USB or built-in camera

### Installation

#### 1. System Preparation

Install required software:

- **Python 3.10**: [Download](https://www.python.org/downloads/)
- **Git**: [Download](https://git-scm.com/downloads)
- **Visual Studio 2022 Runtimes**: [Download](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

#### 2. Clone Repository

```bash
git clone https://github.com/samets81/deep-live-cam.git
cd deep-live-cam
```

#### 3. Download Models

Download the following model files and place them in the `models/` folder:

1. **GFPGANv1.4.pth**: [Download](https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth)
2. **inswapper_128.onnx**: [Download](https://huggingface.co/Patil/inswapper/tree/main)

`models/` folder structure:
```
models/
├── GFPGANv1.4.pth
├── inswapper_128.onnx
└── insightface/        (created automatically)
```

#### 4. Run Installer

```bash
install.bat
```

## 💻 Usage

### Launch Application

```bash
run.bat
```

### Application Interface

1. **Select Photo**: Click "Select Photo" and choose an image with a face for swapping
2. **Select Camera**: Choose a webcam from the dropdown list
3. **Set Resolution**: Select desired resolution (640x480 recommended for start)
4. **Adjust Parameters**:
   - **Face Sharpness**: Controls sharpness of the swapped face
   - **Transition Smoothness**: Manages mask edge blur
   - **Oval Width/Height**: Adjusts swap mask shape
5. **GFPGAN**: Enable for face quality enhancement (reduces FPS)
6. **Start**: Click "Start" to begin processing

## ⚙️ Configuration

### `settings.py` File

Main configuration parameters:

```python
# Default camera resolution
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

# Execution provider
DEFAULT_PROVIDER = "TensorRT"  # or "CUDA"

# Default parameters
DEFAULT_SHARPNESS = 0.4
DEFAULT_MASK_BLUR = 30
DEFAULT_OVAL_WIDTH = 0.45
DEFAULT_OVAL_HEIGHT = 0.55

# TensorRT configuration
TENSORRT_CONFIG = {
    'trt_fp16_enable': True,                    # FP16 for acceleration
    'trt_builder_optimization_level': 3,        # Maximum optimization
    'trt_engine_cache_enable': True,            # Engine caching
    'trt_max_workspace_size': 4294967296,       # 4 GB
}
```

### Performance Optimization

The project includes several optimizations:

- **Mask Caching**: Pre-calculated masks for different face sizes
- **Frame Skipping**: Face detection every N-th frame (configurable)
- **Multi-threading**: Separate threads for capture and processing
- **Frame Queues**: Buffering for smooth processing
- **TensorRT Engine Cache**: Faster subsequent launches

## 🔧 Project Structure

```
face-swapper-tensorrt/
├── app_gpu.py              # Main application
├── ui.py                   # PyQt5 interface
├── settings.py             # Configuration
├── requirements.txt        # Python dependencies
├── run.bat                 # Program launcher
├── install.bat             # Installer
├── models/                 # Models folder
│   ├── GFPGANv1.4.pth
│   ├── inswapper_128.onnx
│   ├── insightface/
│   └── trt_cache/
├── cuda-dll/              # CUDA DLL
├── custom-cv2/            # Custom OpenCV with CUDA support
└── README.md              # Documentation
```

## 📊 Performance

Typical performance on various hardware:

| GPU | Resolution | FPS |
|-----|-----------|-----|
| RTX 5070 ti | 640x480 | ~25 |
| RTX 4060 | 640x480 | ~20 |

*Note: Performance depends on scene complexity and number of faces*

## 🐛 Troubleshooting

### CUDA Issues

```bash
# Check CUDA version
nvcc --version

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### TensorRT Issues

- Ensure cuDNN is properly installed and added to PATH
- Delete TensorRT cache: `models/trt_cache/`
- Try using CUDA provider instead of TensorRT

### Low FPS

- Reduce camera resolution
- Disable GFPGAN
- Increase `DETECT_EVERY_N` in `app_gpu.py`
- Close other GPU applications

### Camera Errors

- Check if camera is not used by another application
- Try another camera from the list
- Restart the application

## 🤝 Acknowledgments

This project uses the following libraries and models:

- **InsightFace**: [GitHub](https://github.com/deepinsight/insightface) - Face analysis and recognition
- **GFPGAN**: [GitHub](https://github.com/TencentARC/GFPGAN) - Face quality enhancement
- **ONNX Runtime**: [Official Site](https://onnxruntime.ai/) - Model execution
- **PyQt5**: GUI framework

## 📄 License

Please note:
- The InsightFace model is intended **for non-commercial research purposes only**
- Review the licenses of used libraries before commercial use

## 📞 Support

If you encounter issues:
1. Check the "Troubleshooting" section
2. Create an Issue on GitHub with detailed problem description
3. Include Python, CUDA, cuDNN versions and system configuration

## 🔄 Updates

Follow project updates:
- New features
- Performance improvements
- Bug fixes
- New model support

---

<p align="center">
  Made with ❤️ for the AI community
</p>

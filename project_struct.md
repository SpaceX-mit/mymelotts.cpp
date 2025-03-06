项目结构
melotts-cpp/
├── CMakeLists.txt          # CMake构建文件
├── Makefile                # 可选的Makefile
├── include/                # 头文件目录
│   ├── melotts.h           # 主要接口定义
│   ├── text_processor.h    # 文本处理组件接口
│   └── OnnxWrapper.hpp     # ONNX模型包装器
├── src/                    # 源文件目录
│   ├── melotts.cpp         # 主要实现
│   ├── text_processor.cpp  # 文本处理实现
│   ├── acoustic_model.cpp  # 声学模型实现
│   ├── vocoder.cpp         # 声码器实现
│   ├── OnnxWrapper.cpp     # ONNX包装器实现
│   └── main.cpp            # 命令行工具入口点
├── examples/               # 示例代码
│   └── simple_tts.cpp      # 简单的TTS示例
├── models/                 # 模型目录
│   ├── acoustic_model.onnx # 声学模型
│   ├── vocoder.onnx        # 声码器模型
│   ├── lexicon.txt         # 词汇表
│   └── phonemes.txt        # 音素表
├── scripts/                # 脚本目录
│   └── export_onnx.py      # 从原始模型导出ONNX模型
└── tests/                  # 测试目录
└── test_tts.cpp        # 单元测试

MeloTTS 转为硬件无关的 C++ 实现

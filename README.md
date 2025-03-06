# MeloTTS C++ 实现指南

本文档介绍如何将 melotts.axera (针对爱芯元智芯片的 MeloTTS 版本) 转换为不依赖特定硬件的纯 C++ 实现版本。

## 项目结构

建议按照以下结构组织项目文件：

```
melotts-cpp/
├── CMakeLists.txt          # CMake构建文件
├── Makefile                # 可选的Makefile
├── include/                # 头文件目录
│   ├── melotts.h           # 主要接口定义
│   └── text_processor.h    # 文本处理组件接口
├── src/                    # 源文件目录
│   ├── melotts.cpp         # 主要实现
│   ├── text_processor.cpp  # 文本处理实现
│   ├── acoustic_model.cpp  # 声学模型实现
│   ├── vocoder.cpp         # 声码器实现
│   └── main.cpp            # 命令行工具入口点
├── examples/               # 示例代码
│   └── simple_tts.cpp      # 简单的TTS示例
├── models/                 # 模型目录
│   ├── acoustic_model.onnx # 声学模型
│   ├── vocoder.onnx        # 声码器模型
│   ├── lexicon.txt         # 词汇表
│   └── phonemes.txt        # 音素表
└── tools/                  # 工具脚本
    └── export_model.py     # 模型导出工具
```

## 准备工作

### 1. 安装依赖

首先，需要安装 ONNX Runtime 和其他必要的库：

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake libsndfile1-dev

# 下载并安装 ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-1.15.1.tgz
tar -xzf onnxruntime-linux-x64-1.15.1.tgz
sudo mkdir -p /usr/local/include/onnxruntime
sudo cp -r onnxruntime-linux-x64-1.15.1/include/* /usr/local/include/onnxruntime/
sudo cp onnxruntime-linux-x64-1.15.1/lib/libonnxruntime* /usr/local/lib/
sudo ldconfig
```

### 2. 从 melotts.axera 提取模型

需要从原始的 melotts.axera 项目中提取 ONNX 模型：

```bash
# 假设原始项目在 ~/AI/tts/melotts/melotts.axera

# 创建模型目录
mkdir -p melotts-cpp/models

# 复制模型文件（需要确认实际路径）
cp ~/AI/tts/melotts/melotts.axera/python/models/*.onnx melotts-cpp/models/
cp ~/AI/tts/melotts/melotts.axera/python/data/*.txt melotts-cpp/models/
```

如果原始项目的模型不是 ONNX 格式，可能需要先转换：

```python
# 创建一个 tools/export_model.py 脚本
import torch
import torch.onnx
import os
import sys
sys.path.append("~/AI/tts/melotts/melotts.axera/python")

# 加载原始模型
from models import AcousticModel, Vocoder

# 加载并导出声学模型
acoustic_model = AcousticModel.from_pretrained("path/to/checkpoint")
dummy_input = (torch.zeros(1, 10, dtype=torch.long), torch.zeros(1, dtype=torch.long), torch.ones(1))
torch.onnx.export(acoustic_model, dummy_input, "models/acoustic_model.onnx", 
                  input_names=["phoneme_ids", "speaker_id", "speed"],
                  output_names=["mel_output"],
                  dynamic_axes={"phoneme_ids": {1: "phoneme_length"}})

# 加载并导出声码器
vocoder = Vocoder.from_pretrained("path/to/checkpoint")
dummy_mel = torch.zeros(1, 80, 100)  # 批次大小, 梅尔带数, 帧数
torch.onnx.export(vocoder, dummy_mel, "models/vocoder.onnx",
                  input_names=["mel_spectrogram"],
                  output_names=["waveform"],
                  dynamic_axes={"mel_spectrogram": {2: "time_length"}})
```

## 编译指南

### 使用 CMake 构建

```bash
# 创建并进入构建目录
mkdir -p build
cd build

# 配置
cmake ..

# 编译
make -j$(nproc)

# 安装（可选）
sudo make install
```

### 使用 Makefile 构建

```bash
# 直接编译
make -j$(nproc)

# 安装（可选）
sudo make install
```

## 使用示例

### 命令行使用

```bash
# 基本使用
./melotts_cli -t "你好，这是一个测试。" -o output.wav

# 指定语言
./melotts_cli -t "Hello, this is a test." -l en -o output_en.wav

# 调整语速
./melotts_cli -t "语速可以调整。" -s 1.2 -o output_fast.wav

# 指定说话人
./melotts_cli -t "不同的说话人有不同的声音。" -sp 1 -o output_speaker1.wav
```

### 作为库使用

```cpp
#include <melotts/melotts.h>
#include <iostream>

int main() {
    try {
        // 初始化 MeloTTS
        melotts::MeloTTS tts("./models");
      
        // 设置参数
        tts.set_speed(1.0f);
        tts.set_speaker_id(0);
      
        // 合成语音
        std::string text = "这是一个C++库调用的示例。";
        std::vector<float> audio = tts.synthesize(text, "zh");
      
        // 保存为WAV文件
        tts.save_wav(audio, "output_from_lib.wav");
      
        std::cout << "成功合成语音！" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
  
    return 0;
}
```

## 关键实现点

1. **文本处理**：

   - 从原始项目中移植文本正则化规则
   - 实现音素转换逻辑
   - 支持中文和英文处理
2. **ONNX模型加载**：

   - 使用ONNX Runtime C++ API加载模型
   - 处理动态输入形状
   - 设置适当的内存分配策略
3. **音频处理**：

   - 实现WAV文件保存功能
   - 处理采样率和位深度
4. **优化考虑**：

   - 使用批处理提高效率
   - 应用流水线并行处理
   - 内存管理优化

## 调试技巧

1. 将处理流程分解为小步骤进行测试
2. 使用简单的示例文本对比Python版本和C++版本的结果
3. 保存中间结果（如音素序列、特征向量）进行比对
4. 使用调试工具如gdb, valgrind等进行内存和性能分析

## 潜在问题和解决方案

1. **模型兼容性**：

   - 确保导出的ONNX模型版本与ONNX Runtime兼容
   - 测试不同输入长度的情况
2. **内存管理**：

   - 长文本可能导致内存问题，考虑分段处理
   - 注意内存泄漏，尤其是在异常处理情况下
3. **多线程安全**：

   - 如果在多线程环境中使用，确保关键组件是线程安全的
4. **错误处理**：

   - 实现全面的错误检查和异常处理
   - 提供有意义的错误消息

## 后续改进建议

1. 支持更多语言和音素表
2. 添加实时流式处理能力
3. 优化内存使用，支持资源受限环境
4. 增加批处理功能
5. 提供更灵活的语音控制参数（音高、音量等）



# 将 MeloTTS 转换为硬件无关的 C++ 实现

我已经为你提供了一个将 melotts.axera 转换为不依赖特定硬件的纯 C++ 实现的完整框架。这个实现使用了 ONNX Runtime 作为推理后端，这样就能在通用 CPU 上运行，而不需要爱芯元智的专用硬件。

## 关键组件概述

1. **核心接口 (melotts.h)**：
   * 定义了主要的 TTS 合成接口
   * 管理文本处理、声学模型和声码器组件
2. **文本处理组件**：
   * 文本正则化：处理数字、标点符号等
   * 音素转换：将文本转换为音素序列
   * 支持中文和英文处理
3. **模型推理**：
   * 声学模型：将音素序列转换为声学特征（梅尔频谱图）
   * 声码器：将声学特征转换为波形
   * 使用 ONNX Runtime C++ API 进行模型推理
4. **构建系统**：
   * 提供了 CMake 和 Makefile 两种构建方式
   * 管理依赖项和安装过程

## 实现步骤

要完成这个转换，你需要：

1. **设置项目结构**：
   * 创建包含目录、源代码目录和模型目录
   * 组织头文件和源文件
2. **安装依赖**：
   * 安装 ONNX Runtime 和其他必要的库
   * 设置构建环境
3. **从原始项目提取模型**：
   * 复制或导出 ONNX 格式的声学模型和声码器模型
   * 获取词汇表和音素表等数据文件
4. **实现代码**：
   * 根据提供的代码框架实现各组件
   * 确保文本处理逻辑与原始项目一致
5. **编译和测试**：
   * 使用 CMake 或 Makefile 构建项目
   * 测试基本功能

## 特别注意事项

1. **原始模型格式**：
   * 如果原始项目使用的不是 ONNX 格式，需要先将模型转换为 ONNX
   * 可以使用 PyTorch 的 `torch.onnx.export` 函数
2. **文本处理兼容性**：
   * 文本处理规则需要与原始项目保持一致，以确保输出质量
   * 可能需要移植一些 Python 处理规则到 C++
3. **内存管理**：
   * C++ 版本需要特别注意内存管理，尤其是在处理长文本时
   * 使用智能指针和适当的作用域管理资源
4. **错误处理**：
   * 实现全面的错误检查和异常处理
   * 提供清晰的错误信息

## 使用说明

一旦实现完成，可以通过命令行工具或作为库进行集成：

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs">bash</div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 text-text-300 hover:bg-bg-200 opacity-60 hover:opacity-100" data-state="closed"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="prismjs code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code class="language-bash"><span class=""><span class="token comment"># 命令行使用示例</span><span class="">
</span></span><span class=""><span class="">./melotts_cli -t </span><span class="token string">"你好，这是语音合成测试。"</span><span class=""> -o output.wav</span></span></code></div></div></div></pre>

或者在 C++ 项目中集成：

<pre><div class="relative flex flex-col rounded-lg"><div class="text-text-300 absolute pl-3 pt-2.5 text-xs">cpp</div><div class="pointer-events-none sticky my-0.5 ml-0.5 flex items-center justify-end px-1.5 py-1 mix-blend-luminosity top-0"><div class="from-bg-300/90 to-bg-300/70 pointer-events-auto rounded-md bg-gradient-to-b p-0.5 backdrop-blur-md"><button class="flex flex-row items-center gap-1 rounded-md p-1 py-0.5 text-xs transition-opacity delay-100 text-text-300 hover:bg-bg-200 opacity-60 hover:opacity-100" data-state="closed"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="currentColor" viewBox="0 0 256 256" class="text-text-500 mr-px -translate-y-[0.5px]"><path d="M200,32H163.74a47.92,47.92,0,0,0-71.48,0H56A16,16,0,0,0,40,48V216a16,16,0,0,0,16,16H200a16,16,0,0,0,16-16V48A16,16,0,0,0,200,32Zm-72,0a32,32,0,0,1,32,32H96A32,32,0,0,1,128,32Zm72,184H56V48H82.75A47.93,47.93,0,0,0,80,64v8a8,8,0,0,0,8,8h80a8,8,0,0,0,8-8V64a47.93,47.93,0,0,0-2.75-16H200Z"></path></svg><span class="text-text-200 pr-0.5">Copy</span></button></div></div><div><div class="prismjs code-block__code !my-0 !rounded-lg !text-sm !leading-relaxed"><code class="language-cpp"><span class=""><span class="token comment">// 库使用示例</span><span class="">
</span></span><span class=""><span class="">melotts</span><span class="token double-colon punctuation">::</span><span class="">MeloTTS </span><span class="token function">tts</span><span class="token punctuation">(</span><span class="token string">"./models"</span><span class="token punctuation">)</span><span class="token punctuation">;</span><span class="">
</span></span><span class=""><span class="">std</span><span class="token double-colon punctuation">::</span><span class="">vector</span><span class="token operator"><</span><span class="token keyword">float</span><span class="token operator">></span><span class=""> audio </span><span class="token operator">=</span><span class=""> tts</span><span class="token punctuation">.</span><span class="token function">synthesize</span><span class="token punctuation">(</span><span class="token string">"你好，这是语音合成测试。"</span><span class="token punctuation">,</span><span class=""> </span><span class="token string">"zh"</span><span class="token punctuation">)</span><span class="token punctuation">;</span><span class="">
</span></span><span class=""><span class="">tts</span><span class="token punctuation">.</span><span class="token function">save_wav</span><span class="token punctuation">(</span><span class="">audio</span><span class="token punctuation">,</span><span class=""> </span><span class="token string">"output.wav"</span><span class="token punctuation">)</span><span class="token punctuation">;</span></span></code></div></div></div></pre>

## 后续改进方向

1. 支持更多语言和音素系统
2. 添加流式处理能力，实现实时合成
3. 优化性能，减少内存占用
4. 增加更多控制参数（音高、情感等）

这个框架为你提供了一个起点，你可以根据实际需求和原始代码结构进行调整和完善。

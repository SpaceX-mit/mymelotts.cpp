// acoustic_model.cpp - 声学模型实现
#include "text_processor.h"
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace melotts {

// 声学模型构造函数
AcousticModel::AcousticModel(const std::string& model_path) {
    // 创建ONNX Runtime环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "AcousticModel");
    env_ = new Ort::Env(std::move(env));
    
    // 会话选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    // 创建会话
    session_ = new Ort::Session(*(static_cast<Ort::Env*>(env_)), model_path.c_str(), session_options);
    
    // 获取模型信息
    Ort::AllocatorWithDefaultOptions allocator;
    
    // 设置输入输出名
    size_t num_input_nodes = static_cast<Ort::Session*>(session_)->GetInputCount();
    size_t num_output_nodes = static_cast<Ort::Session*>(session_)->GetOutputCount();
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = static_cast<Ort::Session*>(session_)->GetInputName(i, allocator);
        input_names_.push_back(input_name);
        allocator.Free(input_name);
    }
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = static_cast<Ort::Session*>(session_)->GetOutputName(i, allocator);
        output_names_.push_back(output_name);
        allocator.Free(output_name);
    }
    
    // 加载音素表
    std::ifstream phoneme_file("phonemes.txt");
    std::string line;
    int idx = 0;
    
    while (std::getline(phoneme_file, line)) {
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        phoneme_to_id_[line] = idx++;
    }
}

// 析构函数
AcousticModel::~AcousticModel() {
    delete static_cast<Ort::Session*>(session_);
    delete static_cast<Ort::Env*>(env_);
}

// 前向推理
std::vector<float> AcousticModel::forward(
    const std::vector<std::string>& phonemes, 
    float speed,
    int speaker_id) 
{
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // 将音素转换为ID
    std::vector<int64_t> phoneme_ids;
    for (const auto& phoneme : phonemes) {
        auto it = phoneme_to_id_.find(phoneme);
        if (it != phoneme_to_id_.end()) {
            phoneme_ids.push_back(it->second);
        } else {
            // 未知音素用0代替
            phoneme_ids.push_back(0);
        }
    }
    
    // 准备输入
    std::vector<int64_t> phoneme_shape = {1, static_cast<int64_t>(phoneme_ids.size())};
    Ort::Value phoneme_input = Ort::Value::CreateTensor<int64_t>(
        memory_info, 
        phoneme_ids.data(), 
        phoneme_ids.size(), 
        phoneme_shape.data(), 
        phoneme_shape.size()
    );
    
    // 准备speaker_id输入
    std::vector<int64_t> speaker_id_value = {static_cast<int64_t>(speaker_id)};
    std::vector<int64_t> speaker_shape = {1};
    Ort::Value speaker_input = Ort::Value::CreateTensor<int64_t>(
        memory_info, 
        speaker_id_value.data(), 
        speaker_id_value.size(), 
        speaker_shape.data(), 
        speaker_shape.size()
    );
    
    // 准备speed输入
    std::vector<float> speed_value = {speed};
    std::vector<int64_t> speed_shape = {1};
    Ort::Value speed_input = Ort::Value::CreateTensor<float>(
        memory_info, 
        speed_value.data(), 
        speed_value.size(), 
        speed_shape.data(), 
        speed_shape.size()
    );
    
    // 收集所有输入
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(phoneme_input));
    inputs.push_back(std::move(speaker_input));
    inputs.push_back(std::move(speed_input));
    
    // 运行推理
    std::vector<const char*> input_names_ptr;
    for (const auto& name : input_names_) {
        input_names_ptr.push_back(name);
    }
    
    std::vector<const char*> output_names_ptr;
    for (const auto& name : output_names_) {
        output_names_ptr.push_back(name);
    }
    
    auto output_tensors = static_cast<Ort::Session*>(session_)->Run(
        Ort::RunOptions{nullptr}, 
        input_names_ptr.data(), 
        inputs.data(), 
        inputs.size(), 
        output_names_ptr.data(), 
        output_names_ptr.size()
    );
    
    // 获取输出
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto output_shape = tensor_info.GetShape();
    size_t output_size = tensor_info.GetElementCount();
    
    // 复制结果
    std::vector<float> result(output_data, output_data + output_size);
    return result;
}

// Vocoder构造函数
Vocoder::Vocoder(const std::string& model_path) {
    // 创建ONNX Runtime环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Vocoder");
    env_ = new Ort::Env(std::move(env));
    
    // 会话选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    // 创建会话
    session_ = new Ort::Session(*(static_cast<Ort::Env*>(env_)), model_path.c_str(), session_options);
    
    // 获取模型信息
    Ort::AllocatorWithDefaultOptions allocator;
    
    // 设置输入输出名
    size_t num_input_nodes = static_cast<Ort::Session*>(session_)->GetInputCount();
    size_t num_output_nodes = static_cast<Ort::Session*>(session_)->GetOutputCount();
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = static_cast<Ort::Session*>(session_)->GetInputName(i, allocator);
        input_names_.push_back(input_name);
        allocator.Free(input_name);
    }
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = static_cast<Ort::Session*>(session_)->GetOutputName(i, allocator);
        output_names_.push_back(output_name);
        allocator.Free(output_name);
    }
}

// 析构函数
Vocoder::~Vocoder() {
    delete static_cast<Ort::Session*>(session_);
    delete static_cast<Ort::Env*>(env_);
}

// 前向推理，将梅尔频谱特征转换为波形
std::vector<float> Vocoder::forward(const std::vector<float>& acoustic_features) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // 计算梅尔频谱图的形状
    int64_t n_mels = 80; // MeloTTS通常使用80个梅尔频带
    int64_t n_frames = acoustic_features.size() / n_mels;
    
    // 准备输入形状
    std::vector<int64_t> features_shape = {1, n_mels, n_frames};
    
    // 创建输入tensor
    Ort::Value features_input = Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(acoustic_features.data()), 
        acoustic_features.size(), 
        features_shape.data(), 
        features_shape.size()
    );
    
    // 运行推理
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(features_input));
    
    std::vector<const char*> input_names_ptr;
    for (const auto& name : input_names_) {
        input_names_ptr.push_back(name);
    }
    
    std::vector<const char*> output_names_ptr;
    for (const auto& name : output_names_) {
        output_names_ptr.push_back(name);
    }
    
    auto output_tensors = static_cast<Ort::Session*>(session_)->Run(
        Ort::RunOptions{nullptr}, 
        input_names_ptr.data(), 
        inputs.data(), 
        inputs.size(), 
        output_names_ptr.data(), 
        output_names_ptr.size()
    );
    
    // 获取输出
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto output_shape = tensor_info.GetShape();
    size_t output_size = tensor_info.GetElementCount();
    
    // 复制结果
    std::vector<float> result(output_data, output_data + output_size);
    return result;
}

} // namespace melotts

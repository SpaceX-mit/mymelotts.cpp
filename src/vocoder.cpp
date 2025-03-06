
// vocoder.cpp - 声码器实现

#include "text_processor.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>

namespace melotts {

// Vocoder构造函数
Vocoder::Vocoder(const std::string& model_path) {
    try {
        // 创建ONNX Runtime环境
        m_ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Vocoder");
        
        // 会话选项
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // 创建会话
        m_session = new Ort::Session(m_ort_env, model_path.c_str(), session_options);
        
        // 获取模型信息
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 设置输入输出名称
        m_input_count = m_session->GetInputCount();
        for (size_t i = 0; i < m_input_count; i++) {
            auto input_name = m_session->GetInputNameAllocated(i, allocator).get();
            m_input_names.push_back(input_name);
        }
        
        m_output_count = m_session->GetOutputCount();
        for (size_t i = 0; i < m_output_count; i++) {
            auto output_name = m_session->GetOutputNameAllocated(i, allocator).get();
            m_output_names.push_back(output_name);
        }
        
        std::cout << "声码器初始化成功: " << model_path << std::endl;
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime Error: ") + e.what());
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("声码器初始化失败: ") + e.what());
    }
}

// 析构函数
Vocoder::~Vocoder() {
    if (m_session) {
        delete m_session;
        m_session = nullptr;
    }
}

// 前向推理
std::vector<float> Vocoder::forward(const std::vector<float>& acoustic_features) {
    try {
        if (acoustic_features.empty()) {
            throw std::invalid_argument("声学特征为空");
        }
        
        // 准备内存信息
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // 假设声学特征的形状是 [batch, n_mels, time]
        // 这里需要根据实际模型调整
        int64_t n_mels = 80; // 通常是80个梅尔频带
        int64_t time_len = acoustic_features.size() / n_mels;
        
        // 准备输入张量
        std::vector<Ort::Value> input_tensors;
        std::vector<int64_t> features_shape = {1, n_mels, time_len}; // [batch, n_mels, time]
        
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, 
            const_cast<float*>(acoustic_features.data()), 
            acoustic_features.size(), 
            features_shape.data(), 
            features_shape.size()
        ));
        
        // 准备输入输出名称
        std::vector<const char*> input_names_ptr;
        for (const auto& name : m_input_names) {
            input_names_ptr.push_back(name.c_str());
        }
        
        std::vector<const char*> output_names_ptr;
        for (const auto& name : m_output_names) {
            output_names_ptr.push_back(name.c_str());
        }
        
        // 运行推理
        auto output_tensors = m_session->Run(
            Ort::RunOptions{nullptr}, 
            input_names_ptr.data(), 
            input_tensors.data(), 
            input_tensors.size(), 
            output_names_ptr.data(), 
            output_names_ptr.size()
        );
        
        // 获取输出波形
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        auto output_shape = tensor_info.GetShape();
        size_t output_size = tensor_info.GetElementCount();
        
        // 复制结果到向量
        std::vector<float> waveform(output_data, output_data + output_size);
        return waveform;
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("声码器推理失败: ") + e.what());
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("声码器推理失败: ") + e.what());
    }
}

} // namespace melotts


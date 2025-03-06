// OnnxWrapper.hpp - 通用ONNX包装器

#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <onnxruntime_cxx_api.h>

class OnnxWrapper {
public:
    OnnxWrapper() = default;
    ~OnnxWrapper() {
        if (m_session) {
            delete m_session;
            m_session = nullptr;
        }
    }

    // 禁用拷贝
    OnnxWrapper(const OnnxWrapper&) = delete;
    OnnxWrapper& operator=(const OnnxWrapper&) = delete;

    // 初始化模型
    int Init(const std::string& model_file) {
        try {
            // 创建ONNX Runtime环境
            m_ort_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OnnxWrapper");
            
            // 会话选项
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // 创建会话
            m_session = new Ort::Session(m_ort_env, model_file.c_str(), session_options);
            
            // 获取模型信息
            Ort::AllocatorWithDefaultOptions allocator;
            
            // 设置输入输出名称
            m_input_num = m_session->GetInputCount();
            for (size_t i = 0; i < m_input_num; i++) {
                auto input_name = m_session->GetInputNameAllocated(i, allocator).get();
                m_input_names.push_back(input_name);
                
                // 获取输入形状
                auto type_info = m_session->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                m_input_shapes.push_back(tensor_info.GetShape());
                
                // 计算输入大小
                size_t elem_size = sizeof(float); // 假设输入是float类型
                size_t total_size = elem_size;
                for (auto& dim : m_input_shapes.back()) {
                    if (dim > 0) {
                        total_size *= static_cast<size_t>(dim);
                    }
                }
                m_input_sizes.push_back(total_size);
            }
            
            m_output_num = m_session->GetOutputCount();
            for (size_t i = 0; i < m_output_num; i++) {
                auto output_name = m_session->GetOutputNameAllocated(i, allocator).get();
                m_output_names.push_back(output_name);
                
                // 获取输出形状
                auto type_info = m_session->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                m_output_shapes.push_back(tensor_info.GetShape());
                
                // 计算输出大小
                size_t elem_size = sizeof(float); // 假设输出是float类型
                size_t total_size = elem_size;
                for (auto& dim : m_output_shapes.back()) {
                    if (dim > 0) {
                        total_size *= static_cast<size_t>(dim);
                    }
                }
                m_output_sizes.push_back(total_size);
            }
            
            // 初始化输入数据存储
            m_input_data.resize(m_input_num, nullptr);
            
            return 0;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
            return -1;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return -1;
        }
    }

    // 推理函数 - 为MeloTTS设计的接口，使用副本避免const_cast
    std::vector<Ort::Value> Run(const std::vector<int>& phone, 
                               const std::vector<int>& tones,
                               const std::vector<int>& langids,
                               std::vector<float>& g,
                               float noise_scale,
                               float noise_scale_w,
                               float length_scale,
                               float sdp_ratio) {
        // 创建可修改的副本
        std::vector<int> phone_copy = phone;
        std::vector<int> tones_copy = tones;
        std::vector<int> langids_copy = langids;
        
        int64_t phonelen = phone_copy.size();
        int64_t toneslen = tones_copy.size();
        int64_t langidslen = langids_copy.size();
         
        std::array<int64_t, 1> phone_dims{phonelen};
        std::array<int64_t, 3> g_dims{1, 256, 1};
        std::array<int64_t, 1> tones_dims{toneslen};
        std::array<int64_t, 1> langids_dims{langidslen};
        std::array<int64_t, 1> noise_scale_dims{1};
        std::array<int64_t, 1> length_scale_dims{1};
        std::array<int64_t, 1> noise_scale_w_dims{1};
        std::array<int64_t, 1> sdp_scale_dims{1};

        const char* input_names[] = {"phone", "tone", "language", "g", "noise_scale", "noise_scale_w", "length_scale", "sdp_ratio"};
        const char* output_names[] = {"z_p", "pronoun_lens", "audio_len"};

        Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> input_vals;
        
        // 使用非const副本，避免const_cast
        input_vals.emplace_back(Ort::Value::CreateTensor<int>(memory_info_handler, phone_copy.data(), phone_copy.size(), phone_dims.data(), phone_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<int>(memory_info_handler, tones_copy.data(), tones_copy.size(), tones_dims.data(), tones_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<int>(memory_info_handler, langids_copy.data(), langids_copy.size(), langids_dims.data(), langids_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, g.data(), g.size(), g_dims.data(), g_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &noise_scale, 1, noise_scale_dims.data(), noise_scale_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &noise_scale_w, 1, noise_scale_w_dims.data(), noise_scale_w_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &length_scale, 1, length_scale_dims.data(), length_scale_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &sdp_ratio, 1, sdp_scale_dims.data(), sdp_scale_dims.size()));

        // 直接返回结果，不存储成员变量中
        return m_session->Run(Ort::RunOptions{nullptr}, input_names, input_vals.data(), input_vals.size(), output_names, m_output_num);
    }

    // 获取输入输出数量
    size_t GetInputCount() const { return m_input_num; }
    size_t GetOutputCount() const { return m_output_num; }
    
    // 获取输入形状
    const std::vector<int64_t>& GetInputShape(int input_idx) const {
        if (input_idx < 0 || input_idx >= static_cast<int>(m_input_shapes.size())) {
            throw std::out_of_range("输入索引超出范围");
        }
        return m_input_shapes[input_idx];
    }
    
    // 获取输入大小
    size_t GetInputSize(int input_idx) const {
        if (input_idx < 0 || input_idx >= static_cast<int>(m_input_sizes.size())) {
            throw std::out_of_range("输入索引超出范围");
        }
        return m_input_sizes[input_idx];
    }
    
    // 获取输出大小
    size_t GetOutputSize(int output_idx) const {
        if (output_idx < 0 || output_idx >= static_cast<int>(m_output_sizes.size())) {
            throw std::out_of_range("输出索引超出范围");
        }
        return m_output_sizes[output_idx];
    }
    
    // 设置输入数据
    void SetInput(const void* data, int input_idx) {
        if (input_idx < 0 || input_idx >= static_cast<int>(m_input_num)) {
            throw std::out_of_range("输入索引超出范围");
        }
        m_input_data[input_idx] = data;
    }
    
    // 同步运行推理
    int RunSync() {
        try {
            // 准备内存信息
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            
            // 准备输入张量
            std::vector<Ort::Value> input_tensors;
            for (size_t i = 0; i < m_input_num; i++) {
                const void* data = m_input_data[i];
                if (!data) {
                    throw std::runtime_error("输入数据未设置: " + std::to_string(i));
                }
                
                input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
                    memory_info, 
                    const_cast<float*>(static_cast<const float*>(data)), 
                    m_input_sizes[i] / sizeof(float), 
                    m_input_shapes[i].data(), 
                    m_input_shapes[i].size()
                ));
            }
            
            // 准备输入输出名称
            std::vector<const char*> input_names;
            for (const auto& name : m_input_names) {
                input_names.push_back(name.c_str());
            }
            
            std::vector<const char*> output_names;
            for (const auto& name : m_output_names) {
                output_names.push_back(name.c_str());
            }
            
            // 运行推理
            m_output_tensors = m_session->Run(
                Ort::RunOptions{nullptr}, 
                input_names.data(), 
                input_tensors.data(), 
                input_tensors.size(), 
                output_names.data(), 
                output_names.size()
            );
            
            return 0;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
            return -1;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return -1;
        }
    }
    
    // 获取输出数据
    void GetOutput(void* dst, int output_idx) {
        if (output_idx < 0 || output_idx >= static_cast<int>(m_output_num)) {
            throw std::out_of_range("输出索引超出范围");
        }
        
        if (m_output_tensors.size() <= static_cast<size_t>(output_idx)) {
            throw std::runtime_error("输出张量未生成");
        }
        
        // 复制输出数据
        float* output_data = m_output_tensors[output_idx].GetTensorMutableData<float>();
        auto tensor_info = m_output_tensors[output_idx].GetTensorTypeAndShapeInfo();
        size_t output_size = tensor_info.GetElementCount() * sizeof(float);
        
        memcpy(dst, output_data, output_size);
    }

private:
    Ort::Env m_ort_env{nullptr};
    Ort::Session* m_session{nullptr};
    size_t m_input_num{0};
    size_t m_output_num{0};
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    
    // 新增成员变量
    std::vector<std::vector<int64_t>> m_input_shapes;
    std::vector<std::vector<int64_t>> m_output_shapes;
    std::vector<size_t> m_input_sizes;
    std::vector<size_t> m_output_sizes;
    std::vector<const void*> m_input_data;
    std::vector<Ort::Value> m_output_tensors;
};
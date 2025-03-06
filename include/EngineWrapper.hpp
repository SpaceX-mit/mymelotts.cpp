
// EngineWrapper.hpp - 通用引擎包装器
// 这个类用于替代原始代码中的爱芯元智专用引擎，使用ONNX Runtime实现

#pragma once

#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

class EngineWrapper {
public:
    EngineWrapper() = default;
    ~EngineWrapper() {
        if (m_session) {
            delete m_session;
            m_session = nullptr;
        }
    }

    // 禁用拷贝
    EngineWrapper(const EngineWrapper&) = delete;
    EngineWrapper& operator=(const EngineWrapper&) = delete;

    // 初始化模型
    int Init(const std::string& model_file) {
        try {
            // 创建ONNX Runtime环境
            m_ort_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "EngineWrapper");
            
            // 会话选项
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // 创建会话
            m_session = new Ort::Session(m_ort_env, model_file.c_str(), session_options);
            
            // 获取模型信息
            Ort::AllocatorWithDefaultOptions allocator;
            
            // 设置输入输出名称
            m_input_count = m_session->GetInputCount();
            for (size_t i = 0; i < m_input_count; i++) {
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
                    total_size *= (dim > 0 ? dim : 1); // 处理动态维度
                }
                m_input_sizes.push_back(total_size);
                
                // 预分配内存
                m_input_tensors.emplace_back();
            }
            
            m_output_count = m_session->GetOutputCount();
            for (size_t i = 0; i < m_output_count; i++) {
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
                    total_size *= (dim > 0 ? dim : 1); // 处理动态维度
                }
                m_output_sizes.push_back(total_size);
            }
            
            return 0;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
            return -1;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return -1;
        }
    }

    // 设置输入数据
    void SetInput(const void* data, int input_idx) {
        if (input_idx < 0 || input_idx >= static_cast<int>(m_input_count)) {
            throw std::out_of_range("输入索引超出范围");
        }
        
        m_input_data[input_idx] = data;
    }

    // 运行推理
    int RunSync() {
        try {
            // 准备内存信息
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            
            // 准备输入张量
            m_input_tensors.clear();
            for (size_t i = 0; i < m_input_count; i++) {
                const void* data = m_input_data[i];
                if (!data) {
                    throw std::runtime_error("输入数据未设置: " + std::to_string(i));
                }
                
                m_input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
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
                m_input_tensors.data(), 
                m_input_tensors.size(), 
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
        if (output_idx < 0 || output_idx >= static_cast<int>(m_output_count)) {
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

    // 获取输入大小
    size_t GetInputSize(int input_idx) const {
        if (input_idx < 0 || input_idx >= static_cast<int>(m_input_count)) {
            throw std::out_of_range("输入索引超出范围");
        }
        return m_input_sizes[input_idx];
    }

    // 获取输出大小
    size_t GetOutputSize(int output_idx) const {
        if (output_idx < 0 || output_idx >= static_cast<int>(m_output_count)) {
            throw std::out_of_range("输出索引超出范围");
        }
        return m_output_sizes[output_idx];
    }
    
    // 获取输入形状
    const std::vector<int64_t>& GetInputShape(int input_idx) const {
        if (input_idx < 0 || input_idx >= static_cast<int>(m_input_count)) {
            throw std::out_of_range("输入索引超出范围");
        }
        return m_input_shapes[input_idx];
    }
    
    // 获取输出形状
    const std::vector<int64_t>& GetOutputShape(int output_idx) const {
        if (output_idx < 0 || output_idx >= static_cast<int>(m_output_count)) {
            throw std::out_of_range("输出索引超出范围");
        }
        return m_output_shapes[output_idx];
    }

private:
    Ort::Env m_ort_env{nullptr};
    Ort::Session* m_session{nullptr};
    
    // 输入输出信息
    size_t m_input_count{0};
    size_t m_output_count{0};
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    
    // 形状和大小信息
    std::vector<std::vector<int64_t>> m_input_shapes;
    std::vector<std::vector<int64_t>> m_output_shapes;
    std::vector<size_t> m_input_sizes;
    std::vector<size_t> m_output_sizes;
    
    // 运行时数据
    std::vector<const void*> m_input_data;
    std::vector<Ort::Value> m_input_tensors;
    std::vector<Ort::Value> m_output_tensors;
};



// OnnxWrapper.hpp - 通用ONNX包装器

#pragma once

#include <vector>
#include <string>
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
            }
            
            m_output_num = m_session->GetOutputCount();
            for (size_t i = 0; i < m_output_num; i++) {
                auto output_name = m_session->GetOutputNameAllocated(i, allocator).get();
                m_output_names.push_back(output_name);
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

    // 推理函数 - 为MeloTTS设计的接口
    std::vector<Ort::Value> Run(std::vector<int>& phone, 
                                std::vector<int>& tones,
                                std::vector<int>& langids,
                                std::vector<float>& g,
                                float noise_scale,
                                float noise_scale_w,
                                float length_scale,
                                float sdp_ratio) {
        int64_t phonelen = phone.size();
        int64_t toneslen = tones.size();
        int64_t langidslen = langids.size();
         
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
        input_vals.emplace_back(Ort::Value::CreateTensor<int>(memory_info_handler, phone.data(), phone.size(), phone_dims.data(), phone_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<int>(memory_info_handler, tones.data(), tones.size(), tones_dims.data(), tones_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<int>(memory_info_handler, langids.data(), langids.size(), langids_dims.data(), langids_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, g.data(), g.size(), g_dims.data(), g_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &noise_scale, 1, noise_scale_dims.data(), noise_scale_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &noise_scale_w, 1, noise_scale_w_dims.data(), noise_scale_w_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &length_scale, 1, length_scale_dims.data(), length_scale_dims.size()));
        input_vals.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, &sdp_ratio, 1, sdp_scale_dims.data(), sdp_scale_dims.size()));

        return m_session->Run(Ort::RunOptions{nullptr}, input_names, input_vals.data(), input_vals.size(), output_names, m_output_num);
    }

    // 获取输入输出数量
    size_t GetInputCount() const { return m_input_num; }
    size_t GetOutputCount() const { return m_output_num; }

private:
    Ort::Env m_ort_env{nullptr};
    Ort::Session* m_session{nullptr};
    size_t m_input_num{0};
    size_t m_output_num{0};
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};


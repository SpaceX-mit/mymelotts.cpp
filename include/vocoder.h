// vocoder.h
#pragma once
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace melotts {

class Vocoder {
public:
    Vocoder(const std::string& model_path);
    ~Vocoder();
    
    std::vector<float> forward(const std::vector<float>& acoustic_features);
    
private:
    Ort::Env m_ort_env;
    Ort::Session* m_session = nullptr;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    size_t m_input_count = 0;
    size_t m_output_count = 0;
};

} // namespace melotts
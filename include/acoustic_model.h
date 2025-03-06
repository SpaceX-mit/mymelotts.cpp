#pragma once

#include <string>
#include <vector>
#include <map>

namespace melotts {

class AcousticModel {
public:
    AcousticModel(const std::string& model_path);
    ~AcousticModel();

    std::vector<float> forward(
        const std::vector<std::string>& phonemes,
        float speed,
        int speaker_id);

private:
    void* env_;    // Ort::Env*
    void* session_;  // Ort::Session*
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::map<std::string, int> phoneme_to_id_;
};

} // namespace melotts
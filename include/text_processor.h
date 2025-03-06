
// text_processor.h - 文本处理组件接口
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>

namespace melotts {

// 文本正则化器
class TextNormalizer {
public:
    TextNormalizer();
    std::string normalize(const std::string& text, const std::string& language);

private:
    std::string normalize_chinese(const std::string& text);
    std::string normalize_english(const std::string& text);
    
    // 中文数字转阿拉伯数字
    std::string cn2an(const std::string& text);
};

// 音素化器
class Phonemizer {
public:
    Phonemizer(const std::string& lexicon_path);
    
    std::vector<std::string> phonemize(const std::string& text, const std::string& language);

private:
    std::vector<std::string> phonemize_chinese(const std::string& text);
    std::vector<std::string> phonemize_english(const std::string& text);

    // 词汇表
    std::unordered_map<std::string, std::vector<std::string>> lexicon_;
};

// 声学模型接口
class AcousticModel {
public:
    AcousticModel(const std::string& model_path);
    ~AcousticModel();

    // 禁用拷贝
    AcousticModel(const AcousticModel&) = delete;
    AcousticModel& operator=(const AcousticModel&) = delete;

    // 前向推理, 输入音素序列, 输出声学特征
    std::vector<float> forward(
        const std::vector<std::string>& phonemes, 
        float speed = 1.0f,
        int speaker_id = 0
    );

private:
    // 加载音素映射
    void LoadPhonemeMap(const std::string& phoneme_file);

    // ONNX Runtime 相关成员
    Ort::Env m_ort_env{nullptr};
    Ort::Session* m_session{nullptr};
    
    // 模型输入输出名称
    size_t m_input_count{0};
    size_t m_output_count{0};
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    
    // 音素到索引的映射
    std::unordered_map<std::string, int> m_phoneme_to_id;
};

// 声码器接口
class Vocoder {
public:
    Vocoder(const std::string& model_path);
    ~Vocoder();

    // 禁用拷贝
    Vocoder(const Vocoder&) = delete;
    Vocoder& operator=(const Vocoder&) = delete;

    // 将声学特征转换为波形
    std::vector<float> forward(const std::vector<float>& acoustic_features);

private:
    // ONNX Runtime 相关成员
    Ort::Env m_ort_env{nullptr};
    Ort::Session* m_session{nullptr};
    
    // 模型输入输出名称
    size_t m_input_count{0};
    size_t m_output_count{0};
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

} // namespace melotts


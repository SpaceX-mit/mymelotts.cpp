
// melotts.cpp - 主类实现

#include "melotts.h"
#include "MeloTTSConfig.h"
#include "Lexicon.hpp"
#include "OnnxWrapper.hpp"
#include "AudioFile.h"
#include "text_processor.h"

#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <future>
#include <chrono>
#include <memory>
#include <sys/time.h>

namespace melotts {

// 获取当前时间（毫秒）
static double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// 在音素序列中插入空白
static std::vector<int> intersperse(const std::vector<int>& lst, int item) {
    std::vector<int> result(lst.size() * 2 + 1, item);
    for (size_t i = 1; i < result.size(); i += 2) {
        result[i] = lst[i / 2];
    }
    return result;
}

// 主类的PIMPL实现
class MeloTTSImpl {
public:
    MeloTTSImpl(const std::string& model_dir) : config_() {
        config_.model_dir = model_dir;
        initialize();
    }
    
    ~MeloTTSImpl() {
        // 智能指针会自动处理资源释放
    }
    
    // 设置参数
    void set_speed(float speed) {
        if (speed <= 0.0f) {
            std::cerr << "警告: 无效的语速值，必须为正数。设置为默认值 (1.0)" << std::endl;
            config_.speed = 1.0f;
        } else {
            config_.speed = speed;
            if (config_.verbose) {
                std::cout << "语速设置为: " << config_.speed << std::endl;
            }
        }
    }
    
    void set_speaker_id(int speaker_id) {
        if (speaker_id < 0) {
            std::cerr << "警告: 无效的说话人ID，必须为非负数。设置为默认值 (0)" << std::endl;
            config_.speaker_id = 0;
        } else {
            config_.speaker_id = speaker_id;
            if (config_.verbose) {
                std::cout << "说话人ID设置为: " << config_.speaker_id << std::endl;
            }
        }
    }
    
    void set_noise_scale(float noise_scale) {
        if (noise_scale < 0.0f || noise_scale > 1.0f) {
            std::cerr << "警告: 无效的噪声比例，必须在0.0到1.0之间。设置为默认值 (0.3)" << std::endl;
            config_.noise_scale = 0.3f;
        } else {
            config_.noise_scale = noise_scale;
            if (config_.verbose) {
                std::cout << "噪声比例设置为: " << config_.noise_scale << std::endl;
            }
        }
    }
    
    void set_config(const MeloTTSConfig& config) {
        config_ = config;
        if (!config_.validate()) {
            std::cerr << "警告: 配置验证失败，使用默认配置" << std::endl;
            config_ = MeloTTSConfig();
        }
        if (config_.verbose) {
            config_.print();
        }
    }
    
    // 主要TTS合成函数
    std::vector<float> synthesize(const std::string& text, const std::string& language) {
        if (text.empty()) {
            throw std::invalid_argument("输入文本不能为空");
        }
        
        // 设置语言
        config_.language = language;
        
        double start, end;
        
        // 步骤1: 文本处理
        start = get_current_time();
        if (config_.verbose) {
            std::cout << "处理文本: " << text << std::endl;
        }
        
        // 使用文本正则化器进行处理
        std::string normalized_text = text_normalizer_->normalize(text, language);
        std::vector<std::string> phoneme_seq = phonemizer_->phonemize(normalized_text, language);
        
        end = get_current_time();
        if (config_.verbose) {
            std::cout << "文本处理耗时: " << (end - start) << " ms" << std::endl;
            std::cout << "音素序列: ";
            for (const auto& p : phoneme_seq) {
                std::cout << p << " ";
            }
            std::cout << std::endl;
        }
        
        // 步骤2: 声学模型推理
        start = get_current_time();
        if (config_.verbose) {
            std::cout << "生成声学特征..." << std::endl;
        }
        
        std::vector<float> acoustic_features = acoustic_model_->forward(
            phoneme_seq, 
            config_.speed, 
            config_.speaker_id
        );
        
        end = get_current_time();
        if (config_.verbose) {
            std::cout << "声学模型推理耗时: " << (end - start) << " ms" << std::endl;
        }
        
        // 步骤3: 声码器推理
        start = get_current_time();
        if (config_.verbose) {
            std::cout << "生成波形..." << std::endl;
        }
        
        std::vector<float> audio = vocoder_->forward(acoustic_features);
        
        end = get_current_time();
        if (config_.verbose) {
            std::cout << "声码器推理耗时: " << (end - start) << " ms" << std::endl;
            std::cout << "生成音频长度: " << audio.size() << " 样本" << std::endl;
        }
        
        return audio;
    }
    
    // 保存为WAV文件
    bool save_wav(const std::vector<float>& audio, const std::string& output_path, int sample_rate = 0) {
        if (audio.empty()) {
            std::cerr << "错误: 音频数据为空，无法保存WAV文件" << std::endl;
            return false;
        }
        
        // 如果未指定采样率，使用配置中的采样率
        if (sample_rate <= 0) {
            sample_rate = config_.sample_rate;
        }
        
        try {
            AudioFile<float> audio_file;
            std::vector<std::vector<float>> audio_data{audio};  // 单声道
            audio_file.setAudioBuffer(audio_data);
            audio_file.setSampleRate(sample_rate);
            
            if (!audio_file.save(output_path)) {
                std::cerr << "保存WAV文件失败: " << output_path << std::endl;
                return false;
            }
            
            if (config_.verbose) {
                std::cout << "WAV文件已保存到: " << output_path << std::endl;
                std::cout << "采样率: " << sample_rate << " Hz" << std::endl;
                std::cout << "持续时间: " << audio.size() * 1.0 / sample_rate << " 秒" << std::endl;
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "保存WAV文件时发生错误: " << e.what() << std::endl;
            return false;
        }
    }
    
private:
    // 初始化组件
    void initialize() {
        try {
            // 创建文本处理组件
            text_normalizer_ = std::make_unique<TextNormalizer>();
            phonemizer_ = std::make_unique<Phonemizer>(config_.model_dir + "/lexicon.txt");
            
            // 创建声学模型
            std::string acoustic_model_path = config_.model_dir + "/acoustic_model.onnx";
            acoustic_model_ = std::make_unique<AcousticModel>(acoustic_model_path);
            
            // 创建声码器
            std::string vocoder_path = config_.model_dir + "/vocoder.onnx";
            vocoder_ = std::make_unique<Vocoder>(vocoder_path);
            
            if (config_.verbose) {
                std::cout << "MeloTTS初始化成功!" << std::endl;
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("MeloTTS初始化失败: " + std::string(e.what()));
        }
    }
    
private:
    MeloTTSConfig config_;
    std::unique_ptr<TextNormalizer> text_normalizer_;
    std::unique_ptr<Phonemizer> phonemizer_;
    std::unique_ptr<AcousticModel> acoustic_model_;
    std::unique_ptr<Vocoder> vocoder_;
};

// MeloTTS 公共接口实现
MeloTTS::MeloTTS(const std::string& model_dir) 
    : pimpl_(std::make_unique<MeloTTSImpl>(model_dir)) {}

MeloTTS::~MeloTTS() = default;

std::vector<float> MeloTTS::synthesize(const std::string& text, const std::string& language) {
    return pimpl_->synthesize(text, language);
}

bool MeloTTS::save_wav(const std::vector<float>& audio, const std::string& output_path, int sample_rate) {
    return pimpl_->save_wav(audio, output_path, sample_rate);
}

void MeloTTS::set_speed(float speed) {
    pimpl_->set_speed(speed);
}

void MeloTTS::set_speaker_id(int speaker_id) {
    pimpl_->set_speaker_id(speaker_id);
}

void MeloTTS::set_noise_scale(float noise_scale) {
    pimpl_->set_noise_scale(noise_scale);
}

void MeloTTS::set_config(const MeloTTSConfig& config) {
    pimpl_->set_config(config);
}

} // namespace melotts


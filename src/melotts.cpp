// melotts.cpp - 完整实现

#include "melotts.h"
#include "MeloTTSConfig.h"
#include "Lexicon.hpp"
#include "OnnxWrapper.hpp"
#include "AudioFile.h"

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
    
    void set_noise_scale_w(float noise_scale_w) {
        if (noise_scale_w < 0.0f || noise_scale_w > 1.0f) {
            std::cerr << "警告: 无效的音素持续时间噪声比例，必须在0.0到1.0之间。设置为默认值 (0.6)" << std::endl;
            config_.noise_scale_w = 0.6f;
        } else {
            config_.noise_scale_w = noise_scale_w;
            if (config_.verbose) {
                std::cout << "音素持续时间噪声比例设置为: " << config_.noise_scale_w << std::endl;
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
        
        // 步骤1: 文本转音素
        start = get_current_time();
        if (config_.verbose) {
            std::cout << "转换文本为音素..." << std::endl;
        }
        
        // 修复C++17结构化绑定，替换为C++14兼容的方式
        auto phonemes_result = text_to_phonemes(text, language);
        auto phones = phonemes_result.first;
        auto tones = phonemes_result.second;
        
        end = get_current_time();
        if (config_.verbose) {
            std::cout << "文本处理耗时: " << (end - start) << " ms" << std::endl;
            std::cout << "音素序列长度: " << phones.size() << std::endl;
        }
        
        // 步骤2: 音素到声学特征
        start = get_current_time();
        if (config_.verbose) {
            std::cout << "生成声学特征..." << std::endl;
        }
        
        auto features = phonemes_to_features(phones, tones);
        
        end = get_current_time();
        if (config_.verbose) {
            std::cout << "声学模型推理耗时: " << (end - start) << " ms" << std::endl;
            std::cout << "特征向量大小: " << features.first.size() << std::endl;
            std::cout << "预期音频长度: " << features.second << " 采样点" << std::endl;
        }
        
        // 步骤3: 声学特征到波形
        start = get_current_time();
        if (config_.verbose) {
            std::cout << "生成波形..." << std::endl;
        }
        
        auto audio = features_to_waveform(features.first, features.second);
        
        end = get_current_time();
        if (config_.verbose) {
            std::cout << "声码器推理耗时: " << (end - start) << " ms" << std::endl;
            std::cout << "生成音频长度: " << audio.size() << " 采样点" << std::endl;
            std::cout << "音频时长: " << audio.size() * 1.0 / config_.sample_rate << " 秒" << std::endl;
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
    
    // 中间API：文本到音素
    std::pair<std::vector<int>, std::vector<int>> text_to_phonemes(const std::string& text, const std::string& language) {
        if (!lexicon_) {
            throw std::runtime_error("词典未初始化");
        }
        
        if (config_.verbose) {
            std::cout << "处理文本: '" << text << "' (语言: " << language << ")" << std::endl;
        }
        
        std::vector<int> phones, tones;
        
        try {
            // 使用词典转换文本
            lexicon_->convert(text, phones, tones);
            
            if (phones.empty()) {
                throw std::runtime_error("文本转换为音素失败: 未能生成音素序列");
            }
            
            if (phones.size() != tones.size()) {
                throw std::runtime_error("音素和声调序列长度不匹配");
            }
            
            // 对原始音素序列进行处理（加入空白）
            phones = intersperse(phones, 0);
            tones = intersperse(tones, 0);
            
            if (config_.verbose) {
                std::cout << "音素转换完成，序列长度: " << phones.size() << std::endl;
            }
            
            return std::make_pair(phones, tones);
        } catch (const std::exception& e) {
            std::cerr << "文本转音素过程中出错: " << e.what() << std::endl;
            throw;
        }
    }
    
    // 中间API：音素到声学特征
    std::pair<std::vector<float>, int> phonemes_to_features(const std::vector<int>& phones, const std::vector<int>& tones) {
        if (!encoder_) {
            throw std::runtime_error("声学模型未初始化");
        }
        
        // 检查输入
        if (phones.empty() || tones.empty() || phones.size() != tones.size()) {
            throw std::invalid_argument("无效的音素或声调序列");
        }
        
        // 准备语言ID
        int lang_id = (config_.language == "zh") ? 3 : 0;  // 3 for Chinese, 0 for English
        std::vector<int> langids(phones.size(), lang_id);
        
        // 准备说话人嵌入
        std::vector<float> g = load_speaker_embedding(config_.speaker_id);
        
        // 推理参数
        float length_scale = 1.0f / config_.speed;
        
        // 创建可修改的副本，不使用原始数据
        std::vector<int> phones_copy = phones;
        std::vector<int> tones_copy = tones;
        std::vector<int> langids_copy = langids;
        
        // 运行声学模型
        auto output = encoder_->Run(phones_copy, tones_copy, langids_copy, g, 
                                 config_.noise_scale, 
                                 config_.noise_scale_w, 
                                 length_scale, 
                                 config_.sdp_ratio);
        
        // 解析输出，修复模板语法
        float* zp_data = output.at(0).template GetTensorMutableData<float>();
        int audio_len = output.at(2).template GetTensorMutableData<int>()[0];
        auto zp_info = output.at(0).GetTensorTypeAndShapeInfo();
        auto zp_shape = zp_info.GetShape();
        
        // 提取声学特征
        size_t feature_size = zp_info.GetElementCount();
        std::vector<float> features(zp_data, zp_data + feature_size);
        
        return std::make_pair(features, audio_len);
    }
    
    // 中间API：声学特征到波形
    std::vector<float> features_to_waveform(const std::vector<float>& features, int audio_len) {
        if (!decoder_) {
            throw std::runtime_error("声码器未初始化");
        }
        
        // 获取说话人嵌入
        std::vector<float> g = load_speaker_embedding(config_.speaker_id);
        
        // 准备声码器输入
        auto zp_shape = decoder_->GetInputShape(0);
        int zp_size = decoder_->GetInputSize(0) / sizeof(float);
        int dec_len = zp_size / zp_shape[1];
        int audio_slice_len = decoder_->GetOutputSize(0) / sizeof(float);
        
        // 计算分段数
        int dec_slice_num = int(std::ceil(zp_shape[2] * 1.0 / dec_len));
        std::vector<float> wavlist;
        
        // 创建特征的完整副本以避免const_cast
        std::vector<float> features_copy = features;
        
        // 分段处理长音频
        for (int i = 0; i < dec_slice_num; i++) {
            // 准备当前段的声学特征
            std::vector<float> zp(zp_size, 0);
            int actual_size = (i + 1) * dec_len < zp_shape[2] ? 
                            dec_len : zp_shape[2] - i * dec_len;
            
            // 复制声学特征数据，使用非const数据
            for (int n = 0; n < zp_shape[1]; n++) {
                float* src = features_copy.data() + n * zp_shape[2] + i * dec_len;
                float* dst = zp.data() + n * dec_len;
                std::copy(src, src + actual_size, dst);
            }
            
            // 设置声码器输入
            decoder_->SetInput(zp.data(), 0);
            decoder_->SetInput(g.data(), 1);
            
            // 运行推理
            if (0 != decoder_->RunSync()) {
                throw std::runtime_error("声码器推理失败");
            }
            
            // 获取输出
            std::vector<float> decoder_output(audio_slice_len);
            decoder_->GetOutput(decoder_output.data(), 0);
            
            // 计算实际输出大小
            actual_size = (i + 1) * audio_slice_len < audio_len ?
                        audio_slice_len : audio_len - i * audio_slice_len;
            
            // 添加到结果
            wavlist.insert(wavlist.end(), decoder_output.begin(),
                        decoder_output.begin() + actual_size);
        }
        
        return wavlist;
    }
    
private:
    // 初始化组件
    void initialize() {
        try {
            // 加载词典
            std::string lexicon_file = config_.model_dir + "/lexicon.txt";
            std::string token_file = config_.model_dir + "/tokens.txt";
            lexicon_ = std::make_unique<Lexicon>(lexicon_file, token_file, config_.verbose);
            
            // 加载声学模型
            std::string encoder_file = config_.model_dir + "/encoder.onnx";
            encoder_ = std::make_unique<OnnxWrapper>();
            if (0 != encoder_->Init(encoder_file)) {
                throw std::runtime_error("声学模型初始化失败: " + encoder_file);
            }
            
            // 加载声码器
            std::string decoder_file = config_.model_dir + "/decoder.onnx";
            decoder_ = std::make_unique<OnnxWrapper>();
            if (0 != decoder_->Init(decoder_file)) {
                throw std::runtime_error("声码器初始化失败: " + decoder_file);
            }
            
            // 加载说话人嵌入
            load_speaker_embeddings();
            
            if (config_.verbose) {
                std::cout << "MeloTTS初始化成功" << std::endl;
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("MeloTTS初始化失败: " + std::string(e.what()));
        }
    }
    
    // 加载说话人嵌入
    void load_speaker_embeddings() {
        // 加载说话人嵌入文件
        std::string g_file = config_.model_dir + "/g.bin";
        FILE* fp = fopen(g_file.c_str(), "rb");
        if (!fp) {
            throw std::runtime_error("无法打开说话人嵌入文件: " + g_file);
        }
        
        // 假设文件包含多个说话人的嵌入向量
        // 每个向量大小为256
        const int embedding_size = 256;
        
        // 获取文件大小
        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        rewind(fp);
        
        // 计算说话人数量
        int num_speakers = file_size / (embedding_size * sizeof(float));
        
        if (num_speakers <= 0) {
            fclose(fp);
            throw std::runtime_error("无效的说话人嵌入文件: " + g_file);
        }
        
        // 读取所有说话人嵌入
        speaker_embeddings_.resize(num_speakers);
        for (int i = 0; i < num_speakers; i++) {
            speaker_embeddings_[i].resize(embedding_size);
            size_t read_size = fread(speaker_embeddings_[i].data(), sizeof(float), embedding_size, fp);
            if (read_size != embedding_size) {
                fclose(fp);
                throw std::runtime_error("读取说话人嵌入失败");
            }
        }
        
        fclose(fp);
        
        if (config_.verbose) {
            std::cout << "加载了 " << num_speakers << " 个说话人嵌入" << std::endl;
        }
    }
    
    // 获取特定说话人的嵌入向量
    std::vector<float> load_speaker_embedding(int speaker_id) {
        // 如果说话人ID超出范围，使用默认说话人
        if (speaker_id < 0 || speaker_id >= static_cast<int>(speaker_embeddings_.size())) {
            std::cerr << "警告: 说话人ID " << speaker_id << " 超出范围，使用默认说话人 (0)" << std::endl;
            speaker_id = 0;
        }
        
        return speaker_embeddings_[speaker_id];
    }
    
private:
    MeloTTSConfig config_;
    std::unique_ptr<Lexicon> lexicon_;
    std::unique_ptr<OnnxWrapper> encoder_;
    std::unique_ptr<OnnxWrapper> decoder_;
    std::vector<std::vector<float>> speaker_embeddings_;
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
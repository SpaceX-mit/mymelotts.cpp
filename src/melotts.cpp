// melotts.cpp - 完整实现（优化版）

// 标准库头文件
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <future>
#include <stdexcept>
#include <sys/time.h>

// 项目头文件
#include "melotts.h"
#include "MeloTTSConfig.h"
#include "Lexicon.hpp"
#include "OnnxWrapper.hpp"
#include "AudioFile.h"

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

// 新增：特征重排序函数，正确处理特征维度转换
static std::vector<float> reshapeFeatures(const std::vector<float>& features, 
                                         int feature_frames, int zp_channels, int dec_len) {
    // 创建结果缓冲区
    std::vector<float> reshaped(zp_channels * dec_len, 0.0f);
    
    // 计算需要处理的帧数
    int frames_to_process = std::min(dec_len, feature_frames);
    
    // 重要：理解原始特征的内存布局
    for (int c = 0; c < zp_channels; c++) {
        for (int f = 0; f < frames_to_process; f++) {
            // 源索引 - 通道优先布局
            int src_idx = c * feature_frames + f;
            
            // 目标索引 - [channels, frames]格式
            int dst_idx = c * dec_len + f;
            
            if (src_idx < static_cast<int>(features.size())) {
                reshaped[dst_idx] = features[src_idx];
            }
        }
    }
    
    return reshaped;
}

// 新增：音频后处理函数，提高音质和清晰度
static std::vector<float> postProcessAudio(const std::vector<float>& audio, int target_len, bool enhance = true) {
    // 1. 裁剪到目标长度
    std::vector<float> result = audio;
    if (result.size() > static_cast<size_t>(target_len)) {
        result.resize(target_len);
    } else if (result.size() < static_cast<size_t>(target_len)) {
        result.resize(target_len, 0.0f);
    }
    
    if (!enhance) return result;
    
    // 2. 音频归一化 - 提高音量并减少失真
    float max_amp = 0.0f;
    for (const auto& sample : result) {
        max_amp = std::max(max_amp, std::abs(sample));
    }
    
    // 避免除以零
    if (max_amp > 0.001f) {
        // 设置目标振幅为0.85（提高音量但避免削波）
        float target_amp = 0.85f;
        float scale = target_amp / max_amp;
        
        for (auto& sample : result) {
            sample *= scale;
            
            // 软削波以避免失真
            if (sample > 0.95f) {
                sample = 0.95f + 0.05f * tanh((sample - 0.95f) / 0.05f);
            } else if (sample < -0.95f) {
                sample = -0.95f + 0.05f * tanh((sample + 0.95f) / 0.05f);
            }
        }
    }
    
    // 3. 简单去噪 - 移除低振幅噪声
    const float noise_gate = 0.01f;
    for (auto& sample : result) {
        if (std::abs(sample) < noise_gate) {
            sample = 0.0f;
        }
    }
    
    return result;
}

class MeloTTSImpl {
public:
    MeloTTSImpl(const std::string& model_dir) : config_() {
        config_.model_dir = model_dir;
        config_.enhance_audio = true; // 默认开启音频增强
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
    
    // 设置音频增强开关
    void enable_audio_enhancement(bool enable) {
        config_.enhance_audio = enable;
        if (config_.verbose) {
            std::cout << "音频增强功能: " << (enable ? "已开启" : "已关闭") << std::endl;
        }
    }
    
    // 主要TTS合成函数
    std::vector<float> synthesize(const std::string& text, const std::string& language) {
        if (text.empty()) {
            throw std::invalid_argument("输入文本不能为空");
        }
        
        // 设置语言
        config_.language = language;
        
        // 优化音频质量的参数调整
        // 暂存原始参数
        float original_noise_scale = config_.noise_scale;
        float original_noise_scale_w = config_.noise_scale_w;
        
        // 优化参数 - 降低噪声以提高清晰度
        config_.noise_scale = 0.1f;      // 降低噪声比例，提高清晰度
        config_.noise_scale_w = 0.3f;    // 降低音素持续时间噪声
        
        double start, end;
        
        // 步骤1: 文本转音素
        start = get_current_time();
        if (config_.verbose) {
            std::cout << "转换文本为音素..." << std::endl;
        }
        
        // 获取音素和声调序列
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
        
        // 恢复原始参数
        config_.noise_scale = original_noise_scale;
        config_.noise_scale_w = original_noise_scale_w;
        
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
            // 检查音频质量
            float signal_power = 0.0f;
            float max_amp = 0.0f;
            int zero_count = 0;
            
            for (const auto& sample : audio) {
                signal_power += sample * sample;
                max_amp = std::max(max_amp, std::abs(sample));
                if (std::abs(sample) < 0.001f) {
                    zero_count++;
                }
            }
            
            signal_power /= audio.size();
            float signal_db = 10.0f * std::log10(signal_power + 1e-10f);
            float zero_percent = 100.0f * zero_count / audio.size();
            
            if (config_.verbose) {
                std::cout << "音频统计信息:" << std::endl;
                std::cout << "  - 信号功率: " << signal_power << " (" << signal_db << " dB)" << std::endl;
                std::cout << "  - 最大振幅: " << max_amp << std::endl;
                std::cout << "  - 静音百分比: " << zero_percent << "%" << std::endl;
            }
            
            // 处理音频 - 根据质量决定是否增强
            std::vector<float> processed_audio;
            
            if (config_.enhance_audio || signal_db < -40.0f || zero_percent > 50.0f || max_amp < 0.1f) {
                if (config_.verbose) {
                    std::cout << "正在对音频进行增强处理..." << std::endl;
                }
                processed_audio = postProcessAudio(audio, audio.size(), true);
            } else {
                processed_audio = audio;
            }
            
            // 保存处理后的音频
            AudioFile<float> audio_file;
            std::vector<std::vector<float>> audio_data{processed_audio};
            audio_file.setAudioBuffer(audio_data);
            audio_file.setSampleRate(sample_rate);
            
            if (!audio_file.save(output_path)) {
                std::cerr << "保存WAV文件失败: " << output_path << std::endl;
                return false;
            }
            
            if (config_.verbose) {
                std::cout << "WAV文件已保存到: " << output_path << std::endl;
                std::cout << "采样率: " << sample_rate << " Hz" << std::endl;
                std::cout << "持续时间: " << processed_audio.size() * 1.0 / sample_rate << " 秒" << std::endl;
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
        
        try {
            // 运行声学模型
            auto output = encoder_->Run(phones, tones, langids, g,
                                     config_.noise_scale, 
                                     config_.noise_scale_w, 
                                     length_scale, 
                                     config_.sdp_ratio);
            
            // 解析输出
            // 检查输出是否有效
            if (output.size() < 3) {
                throw std::runtime_error("声学模型输出不足，预期至少3个输出");
            }
            
            float* zp_data = output.at(0).GetTensorMutableData<float>();
            int* audio_len_data = output.at(2).GetTensorMutableData<int>();
            int audio_len = audio_len_data[0];
            
            // 获取形状信息
            auto zp_info = output.at(0).GetTensorTypeAndShapeInfo();
            auto zp_shape = zp_info.GetShape();
            
            if (config_.verbose) {
                std::cout << "z_p 形状: [";
                for (size_t i = 0; i < zp_shape.size(); i++) {
                    std::cout << zp_shape[i];
                    if (i < zp_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            
            // 提取声学特征并考虑其维度
            size_t feature_size = zp_info.GetElementCount();
            std::vector<float> features(zp_data, zp_data + feature_size);
            
            return std::make_pair(features, audio_len);
        } catch (const Ort::Exception& e) {
            std::cerr << "声学模型推理错误: " << e.what() << std::endl;
            throw std::runtime_error(std::string("声学模型推理失败: ") + e.what());
        }
    }
    
    // 中间API：声学特征到波形 - 优化版
    std::vector<float> features_to_waveform(const std::vector<float>& features, int audio_len) {
        if (!decoder_) {
            throw std::runtime_error("声码器未初始化");
        }
        
        try {
            // 获取说话人嵌入
            std::vector<float> g = load_speaker_embedding(config_.speaker_id);
            
            // 获取声码器输入形状
            auto zp_shape = decoder_->GetInputShape(0);
            
            if (config_.verbose) {
                std::cout << "声码器输入形状: [";
                for (auto& dim : zp_shape) {
                    std::cout << dim << ", ";
                }
                std::cout << "]" << std::endl;
            }
            
            // 检查形状是否有效
            if (zp_shape.size() < 3) {
                throw std::runtime_error("声码器输入需要至少3个维度");
            }
            
            // 提取维度信息
            int zp_batch = (zp_shape[0] > 0) ? zp_shape[0] : 1;
            int zp_channels = (zp_shape[1] > 0) ? zp_shape[1] : 192;
            int dec_len = (zp_shape[2] > 0) ? zp_shape[2] : 128;
            
            if (config_.verbose) {
                std::cout << "声码器输入 - 批次: " << zp_batch 
                         << ", 通道数: " << zp_channels 
                         << ", 帧长度: " << dec_len << std::endl;
            }
            
            // 计算特征帧数
            int feature_frames = features.size() / zp_channels;
            int dec_slice_num = (feature_frames + dec_len - 1) / dec_len;  // 向上取整
            
            if (config_.verbose) {
                std::cout << "特征总帧数: " << feature_frames 
                         << ", 需要分段数: " << dec_slice_num << std::endl;
            }
            
            // 获取输出大小信息
            int audio_slice_len = decoder_->GetOutputSize(0) / sizeof(float);
            
            std::vector<float> wavlist;
            wavlist.reserve(audio_len);  // 预分配内存
            
            // 逐段处理特征
            for (int i = 0; i < dec_slice_num; i++) {
                // 当前段的起始帧和处理帧数
                int start_frame = i * dec_len;
                int frames_to_process = std::min(dec_len, feature_frames - start_frame);
                
                if (frames_to_process <= 0) break;
                
                // 使用专用函数重整特征
                std::vector<float> zp_slice = reshapeFeatures(
                    features, 
                    feature_frames, 
                    zp_channels, 
                    dec_len
                );
                
                // 设置声码器输入
                decoder_->SetInput(zp_slice.data(), 0);
                decoder_->SetInput(g.data(), 1);
                
                // 运行推理
                if (0 != decoder_->RunSync()) {
                    throw std::runtime_error("声码器推理失败");
                }
                
                // 获取输出 - 预分配内存
                std::vector<float> current_audio(audio_slice_len);
                decoder_->GetOutput(current_audio.data(), 0);
                
                // 计算当前段实际输出样本数
                int output_samples = std::min(audio_slice_len, audio_len - static_cast<int>(wavlist.size()));
                
                if (output_samples <= 0) break;
                
                // 将当前段添加到结果
                wavlist.insert(wavlist.end(), 
                              current_audio.begin(), 
                              current_audio.begin() + output_samples);
                
                // 检查是否已生成足够的样本
                if (wavlist.size() >= static_cast<size_t>(audio_len)) {
                    break;
                }
            }
            
            // 裁剪或填充到预期长度
            if (wavlist.size() > static_cast<size_t>(audio_len)) {
                wavlist.resize(audio_len);
            } else if (wavlist.size() < static_cast<size_t>(audio_len)) {
                wavlist.resize(audio_len, 0.0f);  // 填充静音
            }
            
            // 对生成的波形进行后处理
            std::vector<float> processed_audio = postProcessAudio(wavlist, audio_len, config_.enhance_audio);
            
            return processed_audio;
        } catch (const std::exception& e) {
            std::cerr << "声码器推理错误: " << e.what() << std::endl;
            throw std::runtime_error(std::string("声码器推理失败: ") + e.what());
        }
    }
    
    // 模型诊断功能
    void diagnoseModels() {
        std::cout << "开始模型诊断..." << std::endl;
    
        // 检查文件是否存在
        std::string encoder_path = config_.model_dir + "/encoder.onnx";
        std::string decoder_path = config_.model_dir + "/decoder.onnx";
        std::string lexicon_path = config_.model_dir + "/lexicon.txt";
        std::string token_path = config_.model_dir + "/tokens.txt";
        std::string g_path = config_.model_dir + "/g.bin";
        
        std::cout << "检查文件是否存在:" << std::endl;
        bool encoder_exists = std::ifstream(encoder_path).good();
        bool decoder_exists = std::ifstream(decoder_path).good();
        bool lexicon_exists = std::ifstream(lexicon_path).good();
        bool token_exists = std::ifstream(token_path).good();
        bool g_exists = std::ifstream(g_path).good();
        
        std::cout << "  - 声学模型文件 (" << encoder_path << "): " 
                << (encoder_exists ? "存在" : "不存在!") << std::endl;
        std::cout << "  - 声码器文件 (" << decoder_path << "): " 
                << (decoder_exists ? "存在" : "不存在!") << std::endl;
        std::cout << "  - 词典文件 (" << lexicon_path << "): " 
                << (lexicon_exists ? "存在" : "不存在!") << std::endl;
        std::cout << "  - 音素表文件 (" << token_path << "): " 
                << (token_exists ? "存在" : "不存在!") << std::endl;
        std::cout << "  - 说话人嵌入文件 (" << g_path << "): " 
                << (g_exists ? "存在" : "不存在!") << std::endl;
        
        if (!encoder_exists || !decoder_exists || !lexicon_exists || !token_exists || !g_exists) {
            std::cerr << "错误: 部分文件缺失!" << std::endl;
            return;
        }
        
        try {
            std::cout << "\n诊断声学模型..." << std::endl;
            if (encoder_) {
                std::cout << "声学模型已加载，输入输出信息:" << std::endl;
                std::cout << "  - 输入数量: " << encoder_->GetInputCount() << std::endl;
                std::cout << "  - 输出数量: " << encoder_->GetOutputCount() << std::endl;
                
                for (size_t i = 0; i < encoder_->GetInputCount(); i++) {
                    auto shape = encoder_->GetInputShape(i);
                    std::cout << "  - 输入 #" << i << " 形状: [";
                    for (size_t j = 0; j < shape.size(); j++) {
                        std::cout << shape[j];
                        if (j < shape.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
            } else {
                std::cerr << "声学模型未初始化!" << std::endl;
            }
            
            std::cout << "\n诊断声码器..." << std::endl;
            if (decoder_) {
                std::cout << "声码器已加载，输入输出信息:" << std::endl;
                std::cout << "  - 输入数量: " << decoder_->GetInputCount() << std::endl;
                std::cout << "  - 输出数量: " << decoder_->GetOutputCount() << std::endl;
                
                for (size_t i = 0; i < decoder_->GetInputCount(); i++) {
                    auto shape = decoder_->GetInputShape(i);
                    std::cout << "  - 输入 #" << i << " 形状: [";
                    for (size_t j = 0; j < shape.size(); j++) {
                        std::cout << shape[j];
                        if (j < shape.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
            } else {
                std::cerr << "声码器未初始化!" << std::endl;
            }
            
            std::cout << "\n词典和说话人嵌入状态:" << std::endl;
            std::cout << "  - 词典: " << (lexicon_ ? "已加载" : "未加载") << std::endl;
            std::cout << "  - 说话人嵌入: " << (speaker_embeddings_.empty() ? "未加载" : "已加载") << std::endl;
            std::cout << "  - 说话人数量: " << speaker_embeddings_.size() << std::endl;
            
            std::cout << "\n测试简单合成..." << std::endl;
            auto result = text_to_phonemes("测试", "zh");
            std::cout << "  - 音素转换: " << (result.first.empty() ? "失败" : "成功") << std::endl;
            std::cout << "  - 音素数量: " << result.first.size() << std::endl;
            
            // 尝试综合测试生成测试音频
            try {
                std::cout << "\n合成测试音频..." << std::endl;
                // 临时开启详细日志
                bool original_verbose = config_.verbose;
                config_.verbose = true;
                
                // 临时优化质量参数
                float original_noise_scale = config_.noise_scale;
                float original_noise_scale_w = config_.noise_scale_w;
                
                config_.noise_scale = 0.1f;
                config_.noise_scale_w = 0.3f;
                
                // 合成测试音频
                auto audio = synthesize("这是一个测试", "zh");
                
                // 恢复原始参数
                config_.verbose = original_verbose;
                config_.noise_scale = original_noise_scale;
                config_.noise_scale_w = original_noise_scale_w;
                
                // 保存测试音频
                std::string test_file = "test_diagnostic.wav";
                if (save_wav(audio, test_file)) {
                    std::cout << "  - 测试音频已保存到: " << test_file << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "  - 测试音频合成失败: " << e.what() << std::endl;
            }
            
            std::cout << "\n诊断完成。" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "诊断过程中出错: " << e.what() << std::endl;
        }
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

void MeloTTS::diagnoseModels() {
    pimpl_->diagnoseModels();
}

void MeloTTS::enable_audio_enhancement(bool enable) {
    pimpl_->enable_audio_enhancement(enable);
}

} // namespace melotts
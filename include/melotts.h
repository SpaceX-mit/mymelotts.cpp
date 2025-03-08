// melotts.h - MeloTTS API接口

#pragma once

#include <string>
#include <vector>
#include <memory>

namespace melotts {

// 前向声明
struct MeloTTSConfig;
class MeloTTSImpl;

// MeloTTS主类
class MeloTTS {
public:
    // 构造函数，需要指定模型目录
    MeloTTS(const std::string& model_dir);
    
    // 析构函数
    ~MeloTTS();
    
    // 合成语音，返回音频波形数据
    std::vector<float> synthesize(const std::string& text, const std::string& language = "zh");
    
    // 保存为WAV文件
    bool save_wav(const std::vector<float>& audio, const std::string& output_path, int sample_rate = 0);
    
    // 设置语速
    void set_speed(float speed);
    
    // 设置说话人ID
    void set_speaker_id(int speaker_id);
    
    // 设置噪声比例
    void set_noise_scale(float noise_scale);
    
    // 设置音素持续时间噪声比例
    void set_noise_scale_w(float noise_scale_w);
    
    // 设置完整配置
    void set_config(const MeloTTSConfig& config);
    
    // 模型诊断功能
    void diagnoseModels();
    
    // 设置声音质量增强开关
    void enable_audio_enhancement(bool enable);

private:
    // PIMPL模式实现
    std::unique_ptr<MeloTTSImpl> pimpl_;
};

}
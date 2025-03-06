
// melotts.h - 主要接口定义
#pragma once

#include <string>
#include <vector>
#include <memory>

namespace melotts {

// 前置声明
struct MeloTTSConfig;
class MeloTTSImpl;

// MeloTTS 主类
class MeloTTS {
public:
    // 构造函数和析构函数
    MeloTTS(const std::string& model_dir);
    ~MeloTTS();

    // 禁用拷贝
    MeloTTS(const MeloTTS&) = delete;
    MeloTTS& operator=(const MeloTTS&) = delete;

    // 主要TTS接口
    std::vector<float> synthesize(const std::string& text, const std::string& language = "zh");
    
    // 保存音频接口
    bool save_wav(const std::vector<float>& audio, const std::string& output_path, int sample_rate = 0);

    // 设置参数
    void set_speed(float speed);
    void set_speaker_id(int speaker_id);
    void set_noise_scale(float noise_scale);
    void set_config(const MeloTTSConfig& config);

private:
    // PIMPL 实现
    std::unique_ptr<MeloTTSImpl> pimpl_;
};

} // namespace melotts


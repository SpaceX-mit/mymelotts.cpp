
// MeloTTSConfig.h - 配置管理
#pragma once

#include <string>
#include <iostream>

namespace melotts {

// MeloTTS配置结构体
struct MeloTTSConfig {
    // 语速参数（1.0为正常速度，大于1加快，小于1减慢）
    float speed = 1.0f;
    
    // 说话人ID（用于多说话人模型）
    int speaker_id = 0;
    
    // 声音随机性控制（影响合成声音的多样性）
    // 较高的值会产生更多变化，但可能导致不稳定
    float noise_scale = 0.3f;
    
    // 音素持续时间随机性控制
    // 较高的值会产生更多的节奏变化
    float noise_scale_w = 0.6f;
    
    // 停止标记预测比率（影响句子边界的检测）
    float sdp_ratio = 0.2f;
    
    // 输出音频采样率
    int sample_rate = 24000;
    
    // 语言设置（zh: 中文, en: 英文）
    std::string language = "zh";
    
    // 是否使用详细日志
    bool verbose = false;
    
    // 设备设置 (CPU or GPU)
    std::string device = "CPU";
    
    // 模型目录
    std::string model_dir = "./models";
    
    // 转音素相关设置
    bool use_prosody = true;     // 是否使用韵律信息
    bool use_word_segment = true; // 是否使用分词
    
    // 高级参数
    int max_decoder_steps = 4000;  // 解码器最大步数
    int batch_size = 1;            // 批处理大小
    int segment_size = 32;         // 分段大小（用于长音频处理）
    
    // ONNX Runtime 相关设置
    int intra_op_num_threads = 1;           // 内部并行线程数
    int inter_op_num_threads = 1;           // 外部并行线程数
    bool use_deterministic_compute = false; // 是否使用确定性计算
    
    // 添加音频增强开关
    bool enhance_audio = true;  // 默认开启音频增强

    // 校验配置有效性
    bool validate() const {
        // 检查语速范围
        if (speed <= 0.0f) {
            return false;
        }
        
        // 检查噪声比例范围
        if (noise_scale < 0.0f || noise_scale > 1.0f ||
            noise_scale_w < 0.0f || noise_scale_w > 1.0f) {
            return false;
        }
        
        // 检查采样率有效性
        if (sample_rate <= 0) {
            return false;
        }
        
        // 检查语言支持
        if (language != "zh" && language != "en") {
            return false;
        }
        
        return true;
    }
    
    // 打印配置信息
    void print() const {
        std::cout << "MeloTTS 配置:" << std::endl;
        std::cout << " - 语速: " << speed << std::endl;
        std::cout << " - 说话人ID: " << speaker_id << std::endl;
        std::cout << " - 噪声比例: " << noise_scale << std::endl;
        std::cout << " - 音素持续时间噪声比例: " << noise_scale_w << std::endl;
        std::cout << " - SDP比例: " << sdp_ratio << std::endl;
        std::cout << " - 采样率: " << sample_rate << std::endl;
        std::cout << " - 语言: " << language << std::endl;
        std::cout << " - 设备: " << device << std::endl;
        std::cout << " - 模型目录: " << model_dir << std::endl;
    }
};

} // namespace melotts



// example.cpp - MeloTTS 使用示例

#include "melotts.h"
#include "MeloTTSConfig.h"
#include <iostream>
#include <chrono>

// 计时类
class Timer {
public:
    Timer(const std::string& name) 
        : name_(name), 
          start_(std::chrono::high_resolution_clock::now()) 
    {}
    
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << name_ << " 耗时: " << duration << " ms" << std::endl;
    }
    
private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

int main() {
    try {
        std::cout << "MeloTTS C++ 示例程序" << std::endl;
        
        // 初始化MeloTTS
        Timer init_timer("初始化");
        melotts::MeloTTS tts("./models");
        
        // 设置合成参数
        tts.set_speed(1.0f);      // 语速 (1.0 为正常速度)
        tts.set_speaker_id(0);    // 说话人ID
        
        // 要合成的文本
        std::string text = "欢迎使用MeloTTS语音合成系统，这是一个示例。";
        
        // 合成语音
        std::vector<float> audio;
        {
            Timer synth_timer("语音合成");
            audio = tts.synthesize(text, "zh");
        }
        
        // 保存为WAV文件
        {
            Timer save_timer("保存WAV文件");
            if (!tts.save_wav(audio, "example_output.wav", 24000)) {
                std::cerr << "保存WAV文件失败!" << std::endl;
                return 1;
            }
        }
        
        // 显示结果信息
        float duration = static_cast<float>(audio.size()) / 24000.0f;
        std::cout << "合成成功!" << std::endl;
        std::cout << "文本: " << text << std::endl;
        std::cout << "音频长度: " << audio.size() << " 样本" << std::endl;
        std::cout << "持续时间: " << duration << " 秒" << std::endl;
        std::cout << "输出文件: example_output.wav" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}


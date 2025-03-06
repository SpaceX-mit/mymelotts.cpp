// example.cpp - MeloTTS C++ 使用示例

#include "melotts.h"
#include "MeloTTSConfig.h"
#include <iostream>
#include <string>
#include <chrono>

// 计时函数
class Timer {
public:
    Timer(const std::string& name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
    
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << name_ << " 耗时: " << duration << " ms" << std::endl;
    }
    
private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

int main(int argc, char** argv) {
    // 解析命令行参数
    std::string model_dir = "../models";
    std::string text = "爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯片。";
    std::string output_file = "output.wav";
    std::string language = "zh";
    float speed = 1.0f;
    int speaker_id = 0;
    
    // 简单的命令行参数解析
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model-dir") {
            if (i + 1 < argc) model_dir = argv[++i];
        } else if (arg == "-t" || arg == "--text") {
            if (i + 1 < argc) text = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) output_file = argv[++i];
        } else if (arg == "-l" || arg == "--language") {
            if (i + 1 < argc) language = argv[++i];
        } else if (arg == "-s" || arg == "--speed") {
            if (i + 1 < argc) speed = std::stof(argv[++i]);
        } else if (arg == "-sp" || arg == "--speaker") {
            if (i + 1 < argc) speaker_id = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "用法: " << argv[0] << " [选项]" << std::endl;
            std::cout << "选项:" << std::endl;
            std::cout << "  -m, --model-dir DIR    模型目录 (默认: ../models)" << std::endl;
            std::cout << "  -t, --text TEXT        要合成的文本 (默认: 爱芯元智...)" << std::endl;
            std::cout << "  -o, --output FILE      输出WAV文件 (默认: output.wav)" << std::endl;
            std::cout << "  -l, --language LANG    语言代码: zh 或 en (默认: zh)" << std::endl;
            std::cout << "  -s, --speed SPEED      语速 (默认: 1.0)" << std::endl;
            std::cout << "  -sp, --speaker ID      说话人ID (默认: 0)" << std::endl;
            std::cout << "  -h, --help             显示此帮助信息" << std::endl;
            return 0;
        }
    }
    
    try {
        // 打印参数
        std::cout << "模型目录: " << model_dir << std::endl;
        std::cout << "输入文本: " << text << std::endl;
        std::cout << "输出WAV: " << output_file << std::endl;
        std::cout << "语言: " << language << std::endl;
        std::cout << "语速: " << speed << std::endl;
        std::cout << "说话人ID: " << speaker_id << std::endl;
        
        // 初始化MeloTTS
        Timer init_timer("初始化");
        melotts::MeloTTS tts(model_dir);
        
        // 设置参数
        tts.set_speed(speed);
        tts.set_speaker_id(speaker_id);
        
        // 也可以通过配置对象设置参数
        /*
        melotts::MeloTTSConfig config;
        config.speed = speed;
        config.speaker_id = speaker_id;
        config.language = language;
        config.verbose = true;
        tts.set_config(config);
        */
        
        // 合成语音
        std::vector<float> audio;
        {
            Timer synth_timer("语音合成");
            audio = tts.synthesize(text, language);
        }
        
        // 保存为WAV文件
        {
            Timer save_timer("保存WAV");
            if (!tts.save_wav(audio, output_file)) {
                std::cerr << "保存WAV文件失败!" << std::endl;
                return 1;
            }
        }
        
        // 计算音频信息
        float duration = static_cast<float>(audio.size()) / 24000.0f; // 假设采样率为24000Hz
        std::cout << "合成完成!" << std::endl;
        std::cout << "音频长度: " << audio.size() << " 采样点" << std::endl;
        std::cout << "持续时间: " << duration << " 秒" << std::endl;
        std::cout << "输出文件: " << output_file << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

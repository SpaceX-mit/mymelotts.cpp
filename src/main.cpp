
// main.cpp - MeloTTS C++ 命令行工具

#include <string>
#include <iostream>
#include <sys/time.h>
#include "melotts.h"
#include "MeloTTSConfig.h"

// 获取当前时间（毫秒）
static double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// 打印帮助信息
void print_usage(const char* program_name) {
    std::cout << "用法: " << program_name << " [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  -m, --model-dir DIR    模型目录 (默认: ./models)" << std::endl;
    std::cout << "  -t, --text TEXT        要合成的文本 (默认: \"你好，世界！\")" << std::endl;
    std::cout << "  -o, --output FILE      输出WAV文件 (默认: output.wav)" << std::endl;
    std::cout << "  -l, --language LANG    语言代码: zh 或 en (默认: zh)" << std::endl;
    std::cout << "  -s, --speed SPEED      语速 (默认: 1.0)" << std::endl;
    std::cout << "  -sp, --speaker ID      说话人ID (默认: 0)" << std::endl;
    std::cout << "  -r, --sample-rate RATE 采样率 (默认: 24000)" << std::endl;
    std::cout << "  -v, --verbose          显示详细信息" << std::endl;
    std::cout << "  -h, --help             显示此帮助信息" << std::endl;
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    std::string model_dir = "./models";
    std::string text = "你好，世界！";
    std::string output_file = "output.wav";
    std::string language = "zh";
    float speed = 1.0f;
    int speaker_id = 0;
    int sample_rate = 24000;
    bool verbose = false;
    
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
        } else if (arg == "-r" || arg == "--sample-rate") {
            if (i + 1 < argc) sample_rate = std::stoi(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "未知选项: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    try {
        // 初始化配置
        melotts::MeloTTSConfig config;
        config.model_dir = model_dir;
        config.language = language;
        config.speed = speed;
        config.speaker_id = speaker_id;
        config.sample_rate = sample_rate;
        config.verbose = verbose;
        
        if (verbose) {
            std::cout << "MeloTTS 命令行工具" << std::endl;
            config.print();
        }
        
        double start_time, end_time;
        
        // 初始化 MeloTTS
        start_time = get_current_time();
        melotts::MeloTTS tts(model_dir);
        tts.set_config(config);
        end_time = get_current_time();
        
        if (verbose) {
            std::cout << "初始化耗时: " << (end_time - start_time) << " ms" << std::endl;
        }
        
        // 合成语音
        start_time = get_current_time();
        std::vector<float> audio = tts.synthesize(text, language);
        end_time = get_current_time();
        
        if (verbose) {
            std::cout << "合成耗时: " << (end_time - start_time) << " ms" << std::endl;
            std::cout << "生成音频长度: " << audio.size() << " 样本" << std::endl;
            std::cout << "音频时长: " << audio.size() * 1.0 / sample_rate << " 秒" << std::endl;
        }
        
        // 保存为WAV文件
        start_time = get_current_time();
        if (!tts.save_wav(audio, output_file, sample_rate)) {
            std::cerr << "保存WAV文件失败!" << std::endl;
            return 1;
        }
        end_time = get_current_time();
        
        if (verbose) {
            std::cout << "保存耗时: " << (end_time - start_time) << " ms" << std::endl;
        }
        
        std::cout << "合成完成! 音频已保存到: " << output_file << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}


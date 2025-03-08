// main.cpp - MeloTTS C++ 命令行工具

#include <string>
#include <iostream>
#include <sys/time.h>
#include <algorithm>
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

std::string sanitizeFilename(const std::string& text) {
    std::string clean_text;
    clean_text.reserve(text.length());
    
    for (size_t i = 0; i < text.length();) {
        // Get UTF-8 character length
        int char_len = 1;
        unsigned char c = text[i];
        if ((c & 0xE0) == 0xC0) char_len = 2;      // 2-byte UTF-8
        else if ((c & 0xF0) == 0xE0) char_len = 3; // 3-byte UTF-8
        else if ((c & 0xF8) == 0xF0) char_len = 4; // 4-byte UTF-8
        
        if (i + char_len <= text.length()) {
            if (char_len > 1) {
                // Keep UTF-8 characters as is
                clean_text.append(text.substr(i, char_len));
            } else {
                // Replace non-alphanumeric ASCII with underscore
                clean_text += std::isalnum(c) ? c : '_';
            }
        }
        i += char_len;
    }
    return clean_text;
}

std::string makeTestFilename(const std::string& text) {
    // Replace spaces and punctuation with underscores
    std::string clean_text = sanitizeFilename(text);
    
    // Limit filename length
    if (clean_text.length() > 20) {
        clean_text = clean_text.substr(0, 20);
    }
    
    // Add unique suffix using last 6 digits of hash
    size_t hash_val = std::hash<std::string>{}(text);
    std::string hash_suffix = std::to_string(hash_val).substr(0, 6);
    
    return "test_" + clean_text + "_" + hash_suffix + ".wav";
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
    bool verbose = true;
    bool diagnose_mode = false;
    
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
        } else if (arg == "-d" || arg == "--diagnose") {
            diagnose_mode = true; 
        }else if (arg == "-h" || arg == "--help") {
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

        // 诊断模式
        if (diagnose_mode) {
            std::cout << "运行模型诊断..." << std::endl;
            tts.diagnoseModels();
            
            // 添加音素处理测试
            std::cout << "\n测试基本音素处理:" << std::endl;
            std::vector<std::string> test_cases = {
                "你好", "世界", "测试", "Hello", "你好，世界！"
            };
            
            for (const auto& test : test_cases) {
                std::cout << "\n处理测试文本: \"" << test << "\"" << std::endl;
                try {
                    auto audio = tts.synthesize(test, language);
                    std::cout << "合成成功! 音频长度: " << audio.size() << " 样本" << std::endl;
                    
                    // 保存诊断音频
                    std::string test_file = makeTestFilename(test);
                    if (tts.save_wav(audio, test_file)) {
                        std::cout << "测试音频已保存到: " << test_file << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "合成失败: " << e.what() << std::endl;
                }
            }
            
            return 0;
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


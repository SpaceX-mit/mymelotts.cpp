// text_processor.cpp - 文本处理组件实现
#include "text_processor.h"
#include <iostream>
#include <fstream>
#include <regex>
#include <algorithm>
#include <sstream>

namespace melotts {

//==================== TextNormalizer 实现 ====================//

TextNormalizer::TextNormalizer() {
    std::cout << "初始化文本正则化器..." << std::endl;
}

std::string TextNormalizer::normalize(const std::string& text, const std::string& language) {
    if (language == "zh" || language == "zh-CN") {
        return normalize_chinese(text);
    } else if (language == "en" || language == "en-US") {
        return normalize_english(text);
    } else {
        // 默认使用英文处理
        return normalize_english(text);
    }
}

std::string TextNormalizer::normalize_chinese(const std::string& text) {
    // 转换中文数字为阿拉伯数字
    std::string result = cn2an(text);
    
    // 修复：不使用正则表达式，直接遍历替换中文标点
    std::string temp;
    temp.reserve(result.length());
    
    for (size_t i = 0; i < result.length();) {
        // 获取UTF-8字符长度
        int charLen = 1;
        if (i < result.length() && (result[i] & 0xE0) == 0xC0) charLen = 2;
        else if (i < result.length() && (result[i] & 0xF0) == 0xE0) charLen = 3;
        else if (i < result.length() && (result[i] & 0xF8) == 0xF0) charLen = 4;
        
        if (i + charLen <= result.length()) {
            std::string ch = result.substr(i, charLen);
            
            // 检查是否为中文标点
            if (ch == "，" || ch == "。" || ch == "！" || ch == "？" || 
                ch == "；" || ch == "：" || ch == "、" || 
                // 以下是中文引号，括号等
                ch == """ || ch == """ || ch == "'" || ch == "'" || 
                ch == "（" || ch == "）" || ch == "《" || ch == "》") {
                // 将标点替换为空格
                temp += " ";
            } else {
                // 保留原字符
                temp += ch;
            }
        }
        
        i += charLen;
    }
    
    result = temp;
    
    // 处理多余空格
    std::regex space_regex("\\s+");
    result = std::regex_replace(result, space_regex, " ");
    
    // 去除首尾空格
    result = std::regex_replace(result, std::regex("^\\s+|\\s+$"), "");
    
    return result;
}

std::string TextNormalizer::normalize_english(const std::string& text) {
    std::string result = text;
    
    // 转为小写
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    // 替换常见缩写
    std::unordered_map<std::string, std::string> abbr_map = {
        {"mr.", "mister"}, {"mrs.", "missus"}, {"dr.", "doctor"},
        {"st.", "street"}, {"ave.", "avenue"}, {"vs.", "versus"}
    };
    
    for (const auto& pair : abbr_map) {
        std::regex abbr_regex("\\b" + pair.first + "\\b");
        result = std::regex_replace(result, abbr_regex, pair.second);
    }
    
    // 修复：单个数字替换为单词
    static const std::string digit_words[] = {
        "zero", "one", "two", "three", "four", 
        "five", "six", "seven", "eight", "nine"
    };
    
    std::string temp;
    temp.reserve(result.length());
    
    for (size_t i = 0; i < result.length(); ++i) {
        if (i > 0 && i < result.length() - 1 && 
            !std::isalnum(result[i-1]) && !std::isalnum(result[i+1]) && 
            result[i] >= '0' && result[i] <= '9') {
            // 独立的单个数字，转换为单词
            int digit = result[i] - '0';
            temp += digit_words[digit];
        } else {
            // 保留原字符
            temp += result[i];
        }
    }
    
    result = temp;
    
    // 替换标点符号为空格
    std::regex punct_regex("[,.!?;:\\-\\[\\](){}]");
    result = std::regex_replace(result, punct_regex, " ");
    
    // 处理多余空格
    std::regex space_regex("\\s+");
    result = std::regex_replace(result, space_regex, " ");
    
    // 去除首尾空格
    result = std::regex_replace(result, std::regex("^\\s+|\\s+$"), "");
    
    return result;
}

std::string TextNormalizer::cn2an(const std::string& text) {
    // 中文数字映射
    static const std::unordered_map<std::string, int> cn_num = {
        {"零", 0}, {"一", 1}, {"二", 2}, {"三", 3}, {"四", 4},
        {"五", 5}, {"六", 6}, {"七", 7}, {"八", 8}, {"九", 9},
        {"〇", 0}, {"壹", 1}, {"贰", 2}, {"叁", 3}, {"肆", 4},
        {"伍", 5}, {"陆", 6}, {"柒", 7}, {"捌", 8}, {"玖", 9}
    };
    
    static const std::unordered_map<std::string, int> cn_unit = {
        {"十", 10}, {"百", 100}, {"千", 1000}, {"万", 10000},
        {"亿", 100000000}, {"拾", 10}, {"佰", 100}, {"仟", 1000}
    };
    
    std::string result;
    result.reserve(text.length());
    
    size_t i = 0;
    while (i < text.length()) {
        bool processed = false;
        
        // 尝试识别连续的中文数字
        if (i < text.length()) {
            // 检查当前位置是否开始一个中文数字
            std::string current_char;
            int char_len = 1;
            
            if ((text[i] & 0xE0) == 0xC0) char_len = 2;
            else if ((text[i] & 0xF0) == 0xE0) char_len = 3;
            else if ((text[i] & 0xF8) == 0xF0) char_len = 4;
            
            if (i + char_len <= text.length()) {
                current_char = text.substr(i, char_len);
            }
            
            // 检查是否为中文数字或单位
            if (cn_num.find(current_char) != cn_num.end() || 
                cn_unit.find(current_char) != cn_unit.end()) {
                
                // 找到连续的中文数字和单位
                size_t start = i;
                int num = 0;
                int temp = 0;
                
                while (i < text.length()) {
                    // 获取当前字符
                    int char_len = 1;
                    if (i < text.length() && (text[i] & 0xE0) == 0xC0) char_len = 2;
                    else if (i < text.length() && (text[i] & 0xF0) == 0xE0) char_len = 3;
                    else if (i < text.length() && (text[i] & 0xF8) == 0xF0) char_len = 4;
                    
                    if (i + char_len > text.length()) break;
                    
                    std::string ch = text.substr(i, char_len);
                    
                    auto num_it = cn_num.find(ch);
                    auto unit_it = cn_unit.find(ch);
                    
                    if (num_it != cn_num.end()) {
                        // 数字
                        temp = num_it->second;
                        i += char_len;
                    } else if (unit_it != cn_unit.end()) {
                        // 单位
                        int unit = unit_it->second;
                        if (temp == 0) temp = 1;
                        
                        if (unit >= 10000) {
                            // 万或亿
                            num = (num + temp) * (unit / 10000);
                            temp = 0;
                        } else {
                            // 十、百、千
                            temp *= unit;
                            num += temp;
                            temp = 0;
                        }
                        i += char_len;
                    } else {
                        // 不是中文数字或单位，结束
                        break;
                    }
                }
                
                // 处理最后的数字
                num += temp;
                
                // 转换为阿拉伯数字并添加到结果中
                if (i > start) {
                    result += std::to_string(num);
                    processed = true;
                }
            }
        }
        
        // 如果没有处理当前字符，则保留原样
        if (!processed) {
            int char_len = 1;
            if (i < text.length() && (text[i] & 0xE0) == 0xC0) char_len = 2;
            else if (i < text.length() && (text[i] & 0xF0) == 0xE0) char_len = 3;
            else if (i < text.length() && (text[i] & 0xF8) == 0xF0) char_len = 4;
            
            if (i + char_len <= text.length()) {
                result += text.substr(i, char_len);
            }
            i += char_len;
        }
    }
    
    return result;
}

//==================== Phonemizer 实现 ====================//

Phonemizer::Phonemizer(const std::string& lexicon_path) {
    std::cout << "初始化音素转换器，词典路径: " << lexicon_path << std::endl;
    
    // 加载词典
    std::ifstream file(lexicon_path);
    if (!file.is_open()) {
        std::cerr << "无法打开词典文件: " << lexicon_path << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // 去除可能的\r
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        std::istringstream iss(line);
        std::string word;
        iss >> word;
        
        std::vector<std::string> phonemes;
        std::string phoneme;
        while (iss >> phoneme) {
            phonemes.push_back(phoneme);
        }
        
        if (!word.empty() && !phonemes.empty()) {
            lexicon_[word] = phonemes;
        }
    }
    
    std::cout << "词典加载完成，共 " << lexicon_.size() << " 个词条" << std::endl;
}

std::vector<std::string> Phonemizer::phonemize(const std::string& text, const std::string& language) {
    if (language == "zh" || language == "zh-CN") {
        return phonemize_chinese(text);
    } else {
        return phonemize_english(text);
    }
}

std::vector<std::string> Phonemizer::phonemize_chinese(const std::string& text) {
    // 中文音素化处理
    std::vector<std::string> result;
    
    // 将文本按空格分词
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        // 查找词典
        auto it = lexicon_.find(word);
        if (it != lexicon_.end()) {
            // 词典中找到整词
            auto phonemes = it->second;
            result.insert(result.end(), phonemes.begin(), phonemes.end());
        } else {
            // 按字符处理
            for (size_t i = 0; i < word.length();) {
                // 获取UTF-8字符
                int charLen = 1;
                if (i < word.length() && (word[i] & 0xE0) == 0xC0) charLen = 2;
                else if (i < word.length() && (word[i] & 0xF0) == 0xE0) charLen = 3;
                else if (i < word.length() && (word[i] & 0xF8) == 0xF0) charLen = 4;
                
                if (i + charLen <= word.length()) {
                    std::string ch = word.substr(i, charLen);
                    
                    // 查找单字音素
                    auto char_it = lexicon_.find(ch);
                    if (char_it != lexicon_.end()) {
                        auto phonemes = char_it->second;
                        result.insert(result.end(), phonemes.begin(), phonemes.end());
                    } else {
                        // 未知字符，加入特殊音素
                        result.push_back("SP");
                    }
                }
                
                i += charLen;
            }
        }
        
        // 词间加入短停顿
        result.push_back("SP");
    }
    
    // 开始和结束加入静音
    result.insert(result.begin(), "SIL");
    result.push_back("SIL");
    
    return result;
}

std::vector<std::string> Phonemizer::phonemize_english(const std::string& text) {
    // 英文音素化处理
    std::vector<std::string> result;
    
    // 将文本按空格分词
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        // 转小写
        std::transform(word.begin(), word.end(), word.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        
        // 查找词典
        auto it = lexicon_.find(word);
        if (it != lexicon_.end()) {
            // 词典中找到
            auto phonemes = it->second;
            result.insert(result.end(), phonemes.begin(), phonemes.end());
        } else {
            // 使用字母作为回退方案
            // 注意：实际应用中应该使用G2P模型
            for (char c : word) {
                if (c >= 'a' && c <= 'z') {
                    // 字母视为一个音素
                    std::string phoneme(1, c);
                    result.push_back(phoneme);
                }
            }
        }
        
        // 词间加入短停顿
        result.push_back("SP");
    }
    
    // 开始和结束加入静音
    result.insert(result.begin(), "SIL");
    result.push_back("SIL");
    
    return result;
}

} // namespace melotts
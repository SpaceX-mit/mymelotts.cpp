// Lexicon.hpp - 词典和音素处理

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set> 
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

class Lexicon {
public:
    Lexicon(const std::string& lexicon_file, const std::string& token_file) {
        loadLexicon(lexicon_file);
        loadTokens(token_file);
    }
    
    // 将文本转换为音素ID和声调ID
    void convert(const std::string& text, std::vector<int>& phones, std::vector<int>& tones) {
        phones.clear();
        tones.clear();
        
        // 分词处理
        std::vector<std::string> words = segment(text);
        
        for (const auto& word : words) {
            // 查找词典
            if (word.empty()) continue;
            
            if (m_word2phonemes.find(word) != m_word2phonemes.end()) {
                // 词典中存在该词
                const auto& phonemes = m_word2phonemes[word];
                for (const auto& phoneme : phonemes) {
                    if (m_token2id.find(phoneme) != m_token2id.end()) {
                        phones.push_back(m_token2id[phoneme]);
                        // 从音素中提取声调（假设声调信息包含在音素中）
                        int tone = extractTone(phoneme);
                        tones.push_back(tone);
                    } else {
                        std::cerr << "未知音素: " << phoneme << std::endl;
                    }
                }
            } else {
                // 逐字处理
                for (size_t i = 0; i < word.length();) {
                    // 处理UTF-8多字节字符
                    int char_len = 1;
                    if ((word[i] & 0xE0) == 0xC0) char_len = 2;
                    else if ((word[i] & 0xF0) == 0xE0) char_len = 3;
                    else if ((word[i] & 0xF8) == 0xF0) char_len = 4;
                    
                    if (i + char_len <= word.length()) {
                        std::string char_str = word.substr(i, char_len);
                        
                        if (m_word2phonemes.find(char_str) != m_word2phonemes.end()) {
                            // 字符在词典中
                            const auto& phonemes = m_word2phonemes[char_str];
                            for (const auto& phoneme : phonemes) {
                                if (m_token2id.find(phoneme) != m_token2id.end()) {
                                    phones.push_back(m_token2id[phoneme]);
                                    int tone = extractTone(phoneme);
                                    tones.push_back(tone);
                                } else {
                                    std::cerr << "未知音素: " << phoneme << std::endl;
                                }
                            }
                        } else {
                            // 未知字符，使用特殊标记
                            std::cerr << "未知字符: " << char_str << std::endl;
                            if (m_token2id.find("<unk>") != m_token2id.end()) {
                                phones.push_back(m_token2id["<unk>"]);
                                tones.push_back(0);  // 默认声调
                            }
                        }
                    }
                    
                    i += char_len;
                }
            }
        }
    }
    
    // 生成交错音素序列（在每个音素之间插入blank）
    static std::vector<int> intersperse(const std::vector<int>& phones, int blank_id = 0) {
        std::vector<int> result(phones.size() * 2 + 1, blank_id);
        for (size_t i = 1; i < result.size(); i += 2) {
            result[i] = phones[i / 2];
        }
        return result;
    }
    
private:
    void loadLexicon(const std::string& lexicon_file) {
        std::ifstream file(lexicon_file);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开词典文件: " + lexicon_file);
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // 移除Windows风格换行符
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            
            std::istringstream iss(line);
            std::string word;
            iss >> word;
            
            std::vector<std::string> phonemes;
            std::string phoneme;
            while (iss >> phoneme) {
                phonemes.push_back(phoneme);
            }
            
            if (!word.empty() && !phonemes.empty()) {
                m_word2phonemes[word] = phonemes;
            }
        }
        
        std::cout << "加载词典完成，词条数: " << m_word2phonemes.size() << std::endl;
    }
    
    void loadTokens(const std::string& token_file) {
        std::ifstream file(token_file);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开音素表文件: " + token_file);
        }
        
        std::string line;
        int idx = 0;
        while (std::getline(file, line)) {
            // 移除Windows风格换行符
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            
            if (!line.empty()) {
                m_token2id[line] = idx++;
                m_id2token[idx-1] = line;
            }
        }
        
        std::cout << "加载音素表完成，音素数: " << m_token2id.size() << std::endl;
    }
    
    // 简单分词，这里只是根据标点和空格进行分割
    // 实际应用中可能需要使用更复杂的分词算法（如jieba等）
    std::vector<std::string> segment(const std::string& text) {
        std::vector<std::string> words;
        std::string word;
        
        for (size_t i = 0; i < text.length();) {
            // 处理UTF-8多字节字符
            int char_len = 1;
            if ((text[i] & 0xE0) == 0xC0) char_len = 2;
            else if ((text[i] & 0xF0) == 0xE0) char_len = 3;
            else if ((text[i] & 0xF8) == 0xF0) char_len = 4;
            
            if (i + char_len <= text.length()) {
                std::string char_str = text.substr(i, char_len);
                
                // 判断是否是标点或空格
                bool is_punctuation = isPunctuation(char_str);
                
                if (is_punctuation || char_str == " ") {
                    if (!word.empty()) {
                        words.push_back(word);
                        word.clear();
                    }
                    if (is_punctuation) {
                        words.push_back(char_str);  // 将标点也作为单独的token
                    }
                } else {
                    word += char_str;
                }
            }
            
            i += char_len;
        }
        
        if (!word.empty()) {
            words.push_back(word);
        }
        
        return words;
    }
    
    // 判断字符是否是标点
    bool isPunctuation(const std::string& ch) {
        static const std::unordered_set<std::string> punct = {
            "。", "，", "、", "；", "：", "？", "！", "…", 
            """, """, "'", "'", "（", "）", "《", "》", 
            "【", "】", "—", "～", "「", "」",
            ".", ",", ";", ":", "?", "!", "\"", "'", "(", ")", "<", ">",
            "[", "]", "-", "~", "/", "\\"
        };
        
        return punct.find(ch) != punct.end();
    }
    
    // 从音素中提取声调信息
    int extractTone(const std::string& phoneme) {
        // 这里假设声调信息是音素的最后一个字符
        // 具体实现需要根据实际的音素表格式调整
        if (phoneme.empty()) return 0;
        
        char last_char = phoneme[phoneme.length() - 1];
        if (isdigit(last_char)) {
            return last_char - '0';
        }
        
        // 如果没有明确的声调标记，返回默认声调0
        return 0;
    }
    
private:
    std::unordered_map<std::string, std::vector<std::string>> m_word2phonemes;
    std::unordered_map<std::string, int> m_token2id;
    std::unordered_map<int, std::string> m_id2token;
    std::unordered_set<std::string> m_punctuations;
};

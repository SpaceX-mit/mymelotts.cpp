// Lexicon.hpp 优化版
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <regex>

class Lexicon {
public:
    Lexicon(const std::string& lexicon_file, const std::string& token_file, bool verbose = true/*cduan*/) 
        : m_verbose(verbose) {
        loadLexicon(lexicon_file);
        loadTokens(token_file);
        initializePunctuations();
    }
    
    // 将文本转换为音素ID和声调ID
    void convert(const std::string& text, std::vector<int>& phones, std::vector<int>& tones) {
        phones.clear();
        tones.clear();
        
        if (m_verbose) {
            std::cout << "原始文本: " << text << std::endl;
        }
        
        // 先进行文本规范化
        std::string normalized_text = normalizeText(text);
        
        if (m_verbose) {
            std::cout << "规范化文本: " << normalized_text << std::endl;
        }
        
        // 分词处理 - 根据语言类型调用不同的分词方法
        std::vector<std::string> sentences = splitSentence(normalized_text);
        
        if (m_verbose) {
            std::cout << "分句结果 (共" << sentences.size() << "句):" << std::endl;
            for (const auto& s : sentences) {
                std::cout << "  - " << s << std::endl;
            }
        }
        
        for (const auto& sentence : sentences) {
            // 处理每个句子
            std::vector<std::pair<std::string, std::string>> words = segment(sentence);
            
            if (m_verbose) {
                std::cout << "句子分词结果 (共" << words.size() << "词):" << std::endl;
                for (const auto& w : words) {
                    std::cout << "  - " << w.first << " [" << w.second << "]" << std::endl;
                }
            }
            
            for (const auto& [word, pos] : words) {
                if (word.empty()) continue;
                
                // 处理英文单词
                if (isEnglishWord(word)) {
                    processEnglishWord(word, phones, tones);
                    continue;
                }
                
                // 处理中文单词或标点
                if (m_word2phonemes.find(word) != m_word2phonemes.end()) {
                    // 词典中存在该词
                    const auto& phonemes = m_word2phonemes[word];
                    
                    if (m_verbose) {
                        std::cout << "词典匹配: " << word << " -> ";
                        for (const auto& p : phonemes) std::cout << p << " ";
                        std::cout << std::endl;
                    }
                    
                    for (const auto& phoneme : phonemes) {
                        if (m_token2id.find(phoneme) != m_token2id.end()) {
                            phones.push_back(m_token2id[phoneme]);
                            // 从音素中提取声调
                            int tone = extractTone(phoneme);
                            tones.push_back(tone);
                        } else {
                            std::cerr << "未知音素: " << phoneme << std::endl;
                            if (m_token2id.find("UNK") != m_token2id.end()) {
                                phones.push_back(m_token2id["UNK"]);
                                tones.push_back(0);
                            }
                        }
                    }
                } else {
                    // 单字处理
                    processCharByChar(word, phones, tones);
                }
            }
        }
        
        if (m_verbose) {
            std::cout << "最终音素ID序列 (长度=" << phones.size() << "):" << std::endl;
            for (size_t i = 0; i < phones.size(); ++i) {
                if (m_id2token.find(phones[i]) != m_id2token.end()) {
                    std::cout << m_id2token[phones[i]] << "(" << phones[i] << ") ";
                } else {
                    std::cout << "UNK(" << phones[i] << ") ";
                }
            }
            std::cout << std::endl;
            
            std::cout << "最终声调序列 (长度=" << tones.size() << "):" << std::endl;
            for (auto t : tones) std::cout << t << " ";
            std::cout << std::endl;
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
    // 文本规范化
    std::string normalizeText(const std::string& text) {
        // 去除多余空格
        std::string result = text;
        result = std::regex_replace(result, std::regex("\\s+"), " ");
        
        // 替换标点符号
        // 修复Lexicon.hpp中的字符串字面量问题
        const std::unordered_map<std::string, std::string> rep_map = {
            {"：", ","}, {"；", ","}, {"，", ","}, {"。", "."}, 
            {"！", "!"}, {"？", "?"}, {"\n", "."}, {"·", ","}, 
            {"、", ","}, {"...", "…"}, {"$", "."},
            {"\u201C", "'"}, {"\u201D", "'"}, {"\u2018", "'"}, {"\u2019", "'"},  // 修复引号字符
            {"（", "'"}, {"）", "'"}, {"(", "'"}, {")", "'"},
            {"《", "'"}, {"》", "'"}, {"【", "'"}, {"】", "'"},
            {"[", "'"}, {"]", "'"}, {"—", "-"}, {"～", "-"},
            {"~", "-"}, {"「", "'"}, {"」", "'"}
        };
        
        for (const auto& [from, to] : rep_map) {
            size_t pos = 0;
            while ((pos = result.find(from, pos)) != std::string::npos) {
                result.replace(pos, from.length(), to);
                pos += to.length();
            }
        }
        
        // 识别并转换数字
        // 简单起见，这里不做详细处理，实际应用中应使用专门的数字转换库
        
        return result;
    }
    
    // 分句
    std::vector<std::string> splitSentence(const std::string& text, int min_len = 10) {
        // 参考Python的split_sentence函数
        std::vector<std::string> sentences;
        
        // 使用标点符号分割句子
        std::string pattern = "([,.!?;。！？；])";
        std::regex re(pattern);
        
        // 使用正则表达式迭代器分割
        std::sregex_token_iterator iter(text.begin(), text.end(), re, -1);
        std::sregex_token_iterator end;
        
        for (; iter != end; ++iter) {
            std::string sentence = iter->str();
            if (!sentence.empty()) {
                // 清理空白字符
                sentence = std::regex_replace(sentence, std::regex("^\\s+|\\s+$"), "");
                if (!sentence.empty()) {
                    sentences.push_back(sentence);
                }
            }
        }
        
        // 合并短句
        return mergeShortSentences(sentences, min_len);
    }
    
    // 合并短句
    std::vector<std::string> mergeShortSentences(const std::vector<std::string>& sentences, int min_len) {
        if (sentences.empty()) return {};
        
        std::vector<std::string> result;
        std::string current;
        int count_len = 0;
        
        for (size_t i = 0; i < sentences.size(); ++i) {
            current += sentences[i] + " ";
            count_len += sentences[i].length();
            
            if (count_len > min_len || i == sentences.size() - 1) {
                if (!current.empty()) {
                    result.push_back(current);
                    current.clear();
                    count_len = 0;
                }
            }
        }
        
        // 进一步合并非常短的句子
        std::vector<std::string> final_result;
        for (const auto& sent : result) {
            if (!final_result.empty() && sent.length() <= 2) {
                final_result.back() += " " + sent;
            } else {
                final_result.push_back(sent);
            }
        }
        
        // 检查最后一个句子
        if (final_result.size() > 1 && final_result.back().length() <= 2) {
            std::string last = final_result.back();
            final_result.pop_back();
            final_result.back() += " " + last;
        }
        
        return final_result;
    }
    
    // 文本分词
    std::vector<std::pair<std::string, std::string>> segment(const std::string& text) {
        // 这里应该使用更复杂的分词算法，例如集成jieba-cpp
        // 简单起见，我们仍使用基于规则的分词方法，但增加了词性标注
        
        std::vector<std::pair<std::string, std::string>> tokens;
        std::string word;
        
        for (size_t i = 0; i < text.length();) {
            // 处理UTF-8多字节字符
            int char_len = getCharLength(text[i]);
            
            if (i + char_len <= text.length()) {
                std::string char_str = text.substr(i, char_len);
                
                // 判断字符类型
                if (isEnglishWord(char_str)) {
                    // 英文单词，提取完整单词
                    size_t word_end = i;
                    while (word_end < text.length() && isEnglishChar(text[word_end])) {
                        word_end++;
                    }
                    std::string eng_word = text.substr(i, word_end - i);
                    tokens.push_back({eng_word, "eng"});
                    i = word_end;
                    continue;
                } else if (isPunctuation(char_str)) {
                    // 标点符号
                    if (!word.empty()) {
                        tokens.push_back({word, "n"}); // 默认名词
                        word.clear();
                    }
                    tokens.push_back({char_str, "w"}); // 标点词性
                } else if (char_str == " ") {
                    // 空格
                    if (!word.empty()) {
                        tokens.push_back({word, "n"}); // 默认名词
                        word.clear();
                    }
                } else {
                    // 汉字或其他字符
                    word += char_str;
                }
            }
            
            i += char_len;
        }
        
        if (!word.empty()) {
            tokens.push_back({word, "n"}); // 默认名词
        }
        
        // 应用预处理规则：合并特殊词组如"不+词"等
        tokens = premergeForModify(tokens);
        
        return tokens;
    }
    
    // 预处理合并词组（简化版的tone_sandhi.py中的premergeForModify函数）
    std::vector<std::pair<std::string, std::string>> premergeForModify(
            const std::vector<std::pair<std::string, std::string>>& tokens) {
        // 合并 "不" 和其后的词
        auto result = mergeBu(tokens);
        // 合并 "一" 和其后的词
        result = mergeYi(result);
        // 合并重复词
        result = mergeReduplication(result);
        
        return result;
    }
    
    // 合并"不"和后面的词
    std::vector<std::pair<std::string, std::string>> mergeBu(
            const std::vector<std::pair<std::string, std::string>>& tokens) {
        std::vector<std::pair<std::string, std::string>> result;
        std::string last_word = "";
        
        for (const auto& token : tokens) {
            const auto& [word, pos] = token;
            if (last_word == "不") {
                if (!result.empty()) {
                    result.back().first = last_word + word;
                }
            } else if (word != "不") {
                result.push_back(token);
            }
            last_word = word;
        }
        
        // 处理单独的"不"
        if (last_word == "不") {
            result.push_back({"不", "d"});
        }
        
        return result;
    }
    
    // 合并"一"和周围的词
    std::vector<std::pair<std::string, std::string>> mergeYi(
            const std::vector<std::pair<std::string, std::string>>& tokens) {
        std::vector<std::pair<std::string, std::string>> result;
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto& [word, pos] = tokens[i];
            
            // 处理 "词-一-词" 模式的重叠，如"看一看"
            if (i > 0 && i < tokens.size() - 1 && 
                word == "一" && 
                tokens[i-1].first == tokens[i+1].first && 
                tokens[i-1].second == "v") {
                
                if (!result.empty()) {
                    result.back().first = result.back().first + "一" + result.back().first;
                }
                // 跳过处理后面的重复词
                i++;
            }
            // 处理独立的"一"
            else if (word == "一" && i < tokens.size() - 1) {
                // 将"一"与后面的词合并
                if (i+1 < tokens.size()) {
                    result.push_back({word + tokens[i+1].first, tokens[i+1].second});
                    i++;
                } else {
                    result.push_back(tokens[i]);
                }
            }
            else {
                result.push_back(tokens[i]);
            }
        }
        
        return result;
    }
    
    // 合并重复词
    std::vector<std::pair<std::string, std::string>> mergeReduplication(
            const std::vector<std::pair<std::string, std::string>>& tokens) {
        std::vector<std::pair<std::string, std::string>> result;
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            const auto& token = tokens[i];
            
            if (!result.empty() && result.back().first == token.first) {
                // 合并重复词
                result.back().first = result.back().first + token.first;
            } else {
                result.push_back(token);
            }
        }
        
        return result;
    }
    
    // 处理英文单词
    void processEnglishWord(const std::string& word, std::vector<int>& phones, std::vector<int>& tones) {
        // 简单实现，实际应使用英文G2P转换
        if (m_verbose) {
            std::cout << "处理英文单词: " << word << std::endl;
        }
        
        // 假设我们有英文词汇到音素的映射
        if (m_english_dict.find(word) != m_english_dict.end()) {
            const auto& phonemes = m_english_dict[word];
            for (const auto& ph : phonemes) {
                if (m_token2id.find(ph) != m_token2id.end()) {
                    phones.push_back(m_token2id[ph]);
                    tones.push_back(0); // 英文默认声调
                }
            }
        } else {
            // 单字符处理
            for (char c : word) {
                std::string ph_str(1, c);
                if (m_token2id.find(ph_str) != m_token2id.end()) {
                    phones.push_back(m_token2id[ph_str]);
                    tones.push_back(0); // 英文默认声调
                } else if (m_token2id.find("UNK") != m_token2id.end()) {
                    phones.push_back(m_token2id["UNK"]);
                    tones.push_back(0);
                }
            }
        }
    }
    
    // 逐字符处理
    void processCharByChar(const std::string& word, std::vector<int>& phones, std::vector<int>& tones) {
        if (m_verbose) {
            std::cout << "逐字符处理: " << word << std::endl;
        }
        
        std::cout << "cduan 逐字符处理: " << word << std::endl;

        for (size_t i = 0; i < word.length();) {
            int char_len = getCharLength(word[i]);
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
                            if (m_token2id.find("UNK") != m_token2id.end()) {
                                phones.push_back(m_token2id["UNK"]);
                                tones.push_back(0);
                            }
                        }
                    }
                } else {
                    // 未知字符
                    std::cerr << "未知字符: " << char_str << std::endl;
                    if (m_token2id.find("UNK") != m_token2id.end()) {
                        phones.push_back(m_token2id["UNK"]);
                        tones.push_back(0);
                    }
                }
            }
            
            i += char_len;
        }
    }
    
    void loadLexicon(const std::string& lexicon_file) {
        std::ifstream file(lexicon_file);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开词典文件: " + lexicon_file);
        }
        
        std::string line;
        while (std::getline(file, line)) {
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
                
                // 同时为英文单词建立映射
                if (isEnglishWord(word)) {
                    m_english_dict[word] = phonemes;
                }
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
        while (std::getline(file, line)) {
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            
            std::istringstream iss(line);
            std::string token;
            int id;
            
            if (iss >> token >> id) {
                m_token2id[token] = id;
                m_id2token[id] = token;
            }
        }
        
        // 检查是否包含必要的特殊符号
        if (m_token2id.find("_") == m_token2id.end()) {
            std::cerr << "警告: 音素表中缺少必要的填充符号 '_'" << std::endl;
        }
        if (m_token2id.find("UNK") == m_token2id.end()) {
            std::cerr << "警告: 音素表中缺少未知符号 'UNK'" << std::endl;
        }
        
        std::cout << "加载音素表完成，音素数: " << m_token2id.size() << std::endl;
    }
    
    void initializePunctuations() {
        const std::vector<std::string> puncts = {
            "!", "?", "…", ",", ".", "'", "-", "¿", "¡",
            "。", "，", "、", "；", "：", "？", "！", "…", 
            """, """, "'", "'", "（", "）", "《", "》", 
            "【", "】", "—", "～", "「", "」"
        };
        
        for (const auto& p : puncts) {
            m_punctuations.insert(p);
        }
    }
    
    // 获取UTF-8字符长度
    int getCharLength(char first_byte) {
        if ((first_byte & 0x80) == 0) return 1;
        else if ((first_byte & 0xE0) == 0xC0) return 2;
        else if ((first_byte & 0xF0) == 0xE0) return 3;
        else if ((first_byte & 0xF8) == 0xF0) return 4;
        return 1; // 默认
    }
    
    // 判断是否是英文单词
    bool isEnglishWord(const std::string& word) {
        return std::all_of(word.begin(), word.end(), [](char c) {
            return isEnglishChar(c);
        });
    }
    
    // 判断是否是英文字符
    static bool isEnglishChar(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '\'';
    }
    
    // 判断字符是否是标点
    bool isPunctuation(const std::string& ch) {
        return m_punctuations.find(ch) != m_punctuations.end();
    }
    
    // 从音素中提取声调信息
    int extractTone(const std::string& phoneme) {
        if (phoneme.empty()) return 0;
        
        // 首先检查是否有带数字的声调模式
        if (std::isdigit(phoneme.back())) {
            return phoneme.back() - '0';
        }
        
        // 对于某些音素，可能有固定的声调规则
        // 例如，标点符号可能对应特定声调
        if (isPunctuation(phoneme)) {
            return 0; // 标点声调
        }
        
        // 其他情况，返回默认声调
        return 0;
    }
    
private:
    std::unordered_map<std::string, std::vector<std::string>> m_word2phonemes;
    std::unordered_map<std::string, std::vector<std::string>> m_english_dict;
    std::unordered_map<std::string, int> m_token2id;
    std::unordered_map<int, std::string> m_id2token;
    std::unordered_set<std::string> m_punctuations;
    bool m_verbose;
};
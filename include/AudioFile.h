// AudioFile.h - 简化版音频文件处理类

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// 自定义clamp函数（如果不使用C++17）
template<typename T>
T clamp(const T& value, const T& low, const T& high) {
    return std::max(low, std::min(value, high));
}

template <class T>
class AudioFile {
public:
    AudioFile() : sampleRate(44100), bitDepth(16) {}
    
    // 设置音频缓冲区
    void setAudioBuffer(const std::vector<std::vector<T>>& newBuffer) {
        audioData = newBuffer;
    }
    
    // 设置采样率
    void setSampleRate(int newSampleRate) {
        sampleRate = newSampleRate;
    }
    
    // 设置位深度
    void setBitDepth(int newBitDepth) {
        bitDepth = newBitDepth;
    }
    
    // 获取通道数
    int getNumChannels() const {
        return audioData.size();
    }
    
    // 获取每个通道的样本数
    int getNumSamplesPerChannel() const {
        if (audioData.empty()) return 0;
        return audioData[0].size();
    }
    
    // 保存为WAV文件
    bool save(const std::string& filePath) {
        std::ofstream file(filePath, std::ios::binary);
        
        if (!file.is_open()) {
            std::cerr << "无法创建文件: " << filePath << std::endl;
            return false;
        }
        
        int numChannels = getNumChannels();
        int numSamples = getNumSamplesPerChannel();
        
        // 没有音频数据
        if (numChannels == 0 || numSamples == 0) {
            std::cerr << "没有音频数据可保存" << std::endl;
            return false;
        }
        
        // WAV文件头部结构
        struct WavHeader {
            // RIFF头 - 使用字符数组初始化，避免包含隐式空终止符
            char riff[4] = {'R', 'I', 'F', 'F'};
            uint32_t chunk_size;
            char wave[4] = {'W', 'A', 'V', 'E'};
            
            // fmt子块
            char fmt[4] = {'f', 'm', 't', ' '};
            uint32_t fmt_size = 16;
            uint16_t audio_format = 1; // PCM
            uint16_t num_channels;
            uint32_t sample_rate;
            uint32_t byte_rate;
            uint16_t block_align;
            uint16_t bits_per_sample;
            
            // data子块
            char data[4] = {'d', 'a', 't', 'a'};
            uint32_t data_size;
        };
        
        WavHeader header;
        header.num_channels = numChannels;
        header.sample_rate = sampleRate;
        header.bits_per_sample = bitDepth;
        header.block_align = numChannels * bitDepth / 8;
        header.byte_rate = sampleRate * header.block_align;
        header.data_size = numSamples * header.block_align;
        header.chunk_size = 36 + header.data_size;
        
        // 写入头部
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        
        // 将音频样本写入文件
        if (bitDepth == 16) {
            int16_t sample16;
            
            for (int i = 0; i < numSamples; i++) {
                for (int channel = 0; channel < numChannels; channel++) {
                    // 将浮点数样本转换为16位整数
                    T sample = audioData[channel][i];
                    sample = clamp(sample, static_cast<T>(-1.0), static_cast<T>(1.0));
                    sample16 = static_cast<int16_t>(sample * 32767.0);
                    
                    file.write(reinterpret_cast<const char*>(&sample16), 2);
                }
            }
        } else if (bitDepth == 24) {
            for (int i = 0; i < numSamples; i++) {
                for (int channel = 0; channel < numChannels; channel++) {
                    // 将浮点数样本转换为24位整数
                    T sample = audioData[channel][i];
                    sample = clamp(sample, static_cast<T>(-1.0), static_cast<T>(1.0));
                    int32_t sample24 = static_cast<int32_t>(sample * 8388607.0);
                    
                    uint8_t bytes[3];
                    bytes[0] = sample24 & 0xFF;
                    bytes[1] = (sample24 >> 8) & 0xFF;
                    bytes[2] = (sample24 >> 16) & 0xFF;
                    
                    file.write(reinterpret_cast<const char*>(bytes), 3);
                }
            }
        } else if (bitDepth == 32) {
            int32_t sample32;
            
            for (int i = 0; i < numSamples; i++) {
                // 修复循环变量错误：使用 channel 而不是 i
                for (int channel = 0; channel < numChannels; channel++) {
                    // 将浮点数样本转换为32位整数
                    T sample = audioData[channel][i];
                    sample = clamp(sample, static_cast<T>(-1.0), static_cast<T>(1.0));
                    sample32 = static_cast<int32_t>(sample * 2147483647.0);
                    
                    file.write(reinterpret_cast<const char*>(&sample32), 4);
                }
            }
        } else {
            std::cerr << "不支持的位深度: " << bitDepth << std::endl;
            return false;
        }
        
        file.close();
        return true;
    }
    
private:
    std::vector<std::vector<T>> audioData;  // 音频数据 [通道][样本]
    int sampleRate;  // 采样率
    int bitDepth;    // 位深度
};
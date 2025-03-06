
// test_onnx.cpp - ONNX Runtime 测试程序

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    try {
        // 创建 ONNX Runtime 环境
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        
        // 获取可用提供者
        auto providers = Ort::GetAvailableProviders();
        
        std::cout << "ONNX Runtime 安装成功!" << std::endl;
        std::cout << "ONNX Runtime 版本: " << ORT_API_VERSION << std::endl;
        std::cout << "可用提供者数量: " << providers.size() << std::endl;
        
        for (const auto& provider : providers) {
            std::cout << " - " << provider << std::endl;
        }
        
        std::cout << "测试创建会话..." << std::endl;
        
        // 尝试创建一个空会话选项
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        std::cout << "会话选项创建成功!" << std::endl;
        
        // 尝试创建内存信息
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::cout << "内存信息创建成功!" << std::endl;
        
        // 尝试创建一个小型张量
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<int64_t> shape = {2, 2};
        
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            data.data(), 
            data.size(), 
            shape.data(), 
            shape.size()
        );
        
        std::cout << "张量创建成功!" << std::endl;
        
        // 获取张量信息
        auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
        auto tensor_shape = tensor_info.GetShape();
        auto elem_count = tensor_info.GetElementCount();
        
        std::cout << "张量形状: [";
        for (size_t i = 0; i < tensor_shape.size(); i++) {
            std::cout << tensor_shape[i];
            if (i < tensor_shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        
        std::cout << "元素数量: " << elem_count << std::endl;
        
        // 测试成功
        std::cout << "ONNX Runtime 测试成功!" << std::endl;
        return 0;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime 错误: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}


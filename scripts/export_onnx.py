#!/usr/bin/env python3
# 导出 ONNX 模型的脚本

import os
import sys
import argparse
import torch
import torch.onnx
import numpy as np
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="导出 MeloTTS 模型到 ONNX 格式")
    parser.add_argument("--input_dir", type=str, required=True, help="原始模型目录")
    parser.add_argument("--output_dir", type=str, required=True, help="ONNX 模型输出目录")
    parser.add_argument("--device", type=str, default="cpu", help="导出设备 (cpu 或 cuda)")
    return parser.parse_args()

def export_acoustic_model(model, output_path, device="cpu"):
    """导出声学模型到 ONNX 格式"""
    print(f"导出声学模型到 {output_path}...")
    
    # 设置模型为评估模式
    model.eval()
    
    # 准备示例输入
    phoneme_ids = torch.zeros(10, dtype=torch.long).to(device)  # 长度为10的音素序列
    tones = torch.zeros(10, dtype=torch.long).to(device)        # 音调
    language_ids = torch.zeros(10, dtype=torch.long).to(device) # 语言ID
    speaker_embedding = torch.zeros(1, 256, 1).to(device)       # 说话人嵌入
    noise_scale = torch.tensor([0.667], dtype=torch.float32).to(device)    # 噪声比例
    noise_scale_w = torch.tensor([0.8], dtype=torch.float32).to(device)    # 噪声比例W
    length_scale = torch.tensor([1.0], dtype=torch.float32).to(device)     # 长度比例
    sdp_ratio = torch.tensor([0.2], dtype=torch.float32).to(device)        # SDP比例
    
    # 定义输入和输出名称
    input_names = ["phone", "tone", "language", "g", "noise_scale", "noise_scale_w", "length_scale", "sdp_ratio"]
    output_names = ["z_p", "pronoun_lens", "audio_len"]
    
    # 定义动态轴
    dynamic_axes = {
        "phone": {0: "phoneme_length"},
        "tone": {0: "tone_length"},
        "language": {0: "language_length"},
        "z_p": {1: "output_length"}
    }
    
    # 导出模型
    torch.onnx.export(
        model, 
        (phoneme_ids, tones, language_ids, speaker_embedding, noise_scale, noise_scale_w, length_scale, sdp_ratio),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
        verbose=False
    )
    print("声学模型导出成功")

def export_vocoder(model, output_path, device="cpu"):
    """导出声码器到 ONNX 格式"""
    print(f"导出声码器到 {output_path}...")
    
    # 设置模型为评估模式
    model.eval()
    
    # 准备示例输入 (B, n_mels, T)
    mel = torch.zeros(1, 80, 100).to(device)  # 80梅尔频带，100帧
    
    # 定义输入和输出名称
    input_names = ["mel"]
    output_names = ["audio"]
    
    # 定义动态轴
    dynamic_axes = {
        "mel": {2: "time_length"},
        "audio": {1: "audio_length"}
    }
    
    # 导出模型
    torch.onnx.export(
        model, 
        mel,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=13,
        verbose=False
    )
    print("声码器导出成功")

def export_models(args):
    """导出所有模型到 ONNX 格式"""
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 将原始代码目录添加到系统路径
    sys.path.append(os.path.abspath(args.input_dir))
    
    try:
        # 根据具体的模型结构动态导入
        # 注意: 这里的导入路径需要根据原始项目结构进行调整
        try:
            from models.acoustic_model import AcousticModel
            from models.vocoder import Vocoder
        except ImportError:
            print("尝试备选导入路径...")
            # 备选导入路径
            from acoustic_model import AcousticModel
            from vocoder import Vocoder
        
        device = torch.device(args.device)
        
        # 加载声学模型
        acoustic_model_path = os.path.join(args.input_dir, "acoustic_model.pt")
        if os.path.exists(acoustic_model_path):
            acoustic_model = AcousticModel.from_pretrained(acoustic_model_path)
            acoustic_model.to(device)
            export_acoustic_model(
                acoustic_model,
                os.path.join(args.output_dir, "acoustic_model.onnx"),
                device
            )
        else:
            print(f"警告: 声学模型未找到: {acoustic_model_path}")
        
        # 加载声码器模型
        vocoder_path = os.path.join(args.input_dir, "vocoder.pt")
        if os.path.exists(vocoder_path):
            vocoder = Vocoder.from_pretrained(vocoder_path)
            vocoder.to(device)
            export_vocoder(
                vocoder,
                os.path.join(args.output_dir, "vocoder.onnx"),
                device
            )
        else:
            print(f"警告: 声码器模型未找到: {vocoder_path}")
        
        # 复制其他必要文件
        for file_name in ["lexicon.txt", "phonemes.txt"]:
            src_path = os.path.join(args.input_dir, file_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, os.path.join(args.output_dir, file_name))
                print(f"复制文件: {file_name} 到输出目录")
            else:
                print(f"警告: 文件未找到: {src_path}")
        
    except Exception as e:
        print(f"导出模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    args = parse_args()
    if export_models(args):
        print("\n导出完成!")
        print(f"ONNX 模型已保存到: {args.output_dir}")
    else:
        print("\n导出失败.")
        sys.exit(1)


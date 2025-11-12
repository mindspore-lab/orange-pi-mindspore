# Image Feature Extraction

## 1. 任务简介

本任务展示了如何在 Orange Pi Alpro 开发板上运行图像特征提取（Image Feature Extraction）任务。选用 nvidia/segformer-b0-finetuned-ade-512-512 作为基础模型，进行图像特征提取任务，主要用于图像分类、目标检测等相关任务。

## 2. 环境要求

- **硬件**: Orange Pi Alpro (20T24G)
- **操作系统**: Ubuntu 镜像
- **CANN 版本**: 8.1.RC1
- **Python**: 3.9
- **MindSpore**: 2.6.0 (Ascend 版本)
- **MindSpore NLP**: 0.4.1

## 3. 模型信息

- **模型名称**: nvidia/segformer-b0-finetuned-ade-512-512
- **下载地址**: [https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
- **参数数目**: 约 3.75M 
- **模型文件大小**: < 400 MB
- **用途**: 使用原始模型进行语义分割
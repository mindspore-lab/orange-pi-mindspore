# Image Segmentation

## 1. 任务简介
本任务展示了如何在 Orange Pi Alpro 开发板上运行图像分割（Image Segmentation）任务。选用 **facebook/sam-vit-base** 作为基础模型，进行图像分割任务，主要用于对图像中的对象进行精确的像素级分割，支持任意对象的零样本分割。

## 2. 环境要求
硬件: Orange Pi Alpro (20T24G)  
操作系统: Ubuntu 镜像  
CANN 版本: 8.0.0.beta1  
Python: 3.9  
MindSpore: 2.6.0 (Ascend 版本)  
MindSpore NLP: 0.4.1  

## 3. 模型信息
**模型名称**: facebook/sam-vit-base  
**下载地址**: https://huggingface.co/facebook/sam-vit-base  
**参数数目**: 约 93.7M  
**模型文件大小**: 约 375 MB  
**用途**: 使用原始模型进行任意对象的零样本图像分割，支持点、框等多种提示方式
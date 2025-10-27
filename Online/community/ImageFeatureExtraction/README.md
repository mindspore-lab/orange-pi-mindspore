
# SegFormer Semantic Segmentation (MindSpore 2.6.0 / MindNLP 0.4.1) — OrangePi AIpro (FP16)

- **设备/环境**：Ascend 310B（OrangePi AIpro 20T24G），CANN 8.1RC1，MindSpore 2.6.0，MindNLP 0.4.1，Python 3.9  
- **模型**：`nvidia/segformer-b0-finetuned-ade-512-512`（通过 HF 镜像自动下载）  
- **数据集**：ADE20K（KaggleHub 自动下载）  
- **精度/算子**：全链路 FP16；推理图中移除了 BatchNorm（替换为 Identity），避免 310B 上 BN 内核不支持。

## 使用方法
1. 在 AIpro 上安装必要依赖（确保 CANN/MindSpore/MindNLP 版本与上文一致）。  
2. 依次运行：
   - 数据集下载（python data.py）
   - 评测（python Image_Feature_Extraction.py）
3. 输出：
   - `打印 mIoU / Pixel Acc（在 validation 集上）

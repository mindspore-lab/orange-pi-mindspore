# 越南语图文对搜索

​	基于 MindNLP 框架，使用 XLM-RoBERTa 模型，在香橙派 AIpro 开发板（ Ascend310B NPU，20T24G）上实现越南语图像内容语义搜索。对万卷·丝路越南语图文对数据集建立向量索引，输入越南语查询文本，检索语义最相关的图像及其文本描述。

## 介绍

​	基于香橙派 AIpro 边缘计算硬件的低功耗、高算力特性，结合 MindNLP 框架的灵活部署能力，实现越南语图像内容语义搜索。

## 模型准备

**模型名称：** `FacebookAI/xlm-roberta-base`

**下载地址：** https://huggingface.co/FacebookAI/xlm-roberta-base

## 数据集准备

万卷·丝路 越南语图文对数据集： https://opendatalab.com/OpenDataLab/WanJuanSiLu2O/explore/main

## 环境准备

| 组件       | 版本                                  |
| ---------- | ------------------------------------- |
| CANN       | 8.1.RC1                               |
| MindNLP    | 0.4.1                                 |
| MindSpore  | 2.5.0                                 |
| Python     | 3.9                                   |
| 开发板型号 | Orange Pi AIpro（Ascend 310，20T24G） |

## 快速开始

顺次执行各个cell，最后在query输入需要查找的图像的越南语描述文本即可。 

#### Demo 示例

输入：điện thoại thông minh màu xanh đặt trên bàn

输出：

```
查询: điện thoại thông minh màu xanh đặt trên bàn
=================================================================

Top1  相似度=0.9997  [文化类]
描述: Hướng dẫn cách chụp màn hình Apple Watch nh...
图像: https://cdn-i.vtcnews.vn/resize/th/upload/2023/04/26/chup-man-hinh-apple-watch-1-14395627.jpeg

Top2  相似度=0.9996  [场景类]
描述: Lỗ hổng an ninh năng lượng ...
图像: https://cdn.vietnambiz.vn/2019/5/18/photo-2-155815466312795396684.jpg

Top3  相似度=0.9996  [场景类]
描述: Tài trợ kinh phí để báo chí chống gỗ lậu...
图像: https://thoibaotaichinhvietnam.vn/stores/news_dataimages/thoibaotaichinhvietnamvn/122015/16/16/medium/gl120210814092305.6459560.jpg?151216042100

Top4  相似度=0.9996  [文化类]
描述: gương mặt thân quen...
图像: https://cdn-i.vtcnews.vn/files/f2/2014/06/14/truc-tiep-chung-ket-guong-mat-than-quen-9.jpg

Top5  相似度=0.9996  [文化类]
描述: cha tự long...
图像: https://cdn-i.vtcnews.vn/files/f2/2015/07/11/tu-long-va-10-dieu-khan-gia-it-biet-1.jpg

Top6  相似度=0.9996  [综合类]
描述: Những loại lá bình thường bỗng thành hàng hot trên trang thương mại điện tử  ...
图像: https://cdn-i.vtcnews.vn/resize/th/upload/2021/06/23/la-lot-06450822.jpg
```

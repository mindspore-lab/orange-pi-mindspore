# ros2_robot_arm
[【开源实习】智能机械臂基于香橙派和MindSpore的深度学习AI推理功能和机械臂的二次开发](https://gitee.com/mindspore/community/issues/ICJ8TX)

### 任务信息

- **Issue**: [ICJ8TX](https://gitee.com/mindspore/community/issues/ICJ8TX)
- **任务类型**: `香橙派开发板应用开发`
- **开发者**: [shenyidong2026](https://github.com/22-Pig)

---

### 项目说明
- 本项目基于 Orange Pi AI Pro（香橙派）与昇腾AI开发板，结合 MindSpore 与 YOLO 模型，实现机械臂对任意位置积木的检测、分拣与堆叠功能。文档和示例以 Jupyter Notebook 格式提供，包含每步执行结果（已在笔记本中保留）。

---

### 主要特性
- 相机采集 + YOLOv5 推理（OM 模型）用于积木检测与定位
- 透视变换与标定（十字标定板）保证像素坐标到机械臂坐标的变换
- 机械臂控制（串口/扩展板）实现抓取、转移与堆叠
- 包含校准（GUI）流程、参数微调（offset）与运行示例 GIF

---

### 硬件与软件要求

| 类别 | 要求 |
|---|---|
| 硬件 | 昇腾AI开发板、Orange Pi AI Pro、机械臂套件、USB 摄像头 |
| 开发板镜像 | Ubuntu |
| CANN | 8.1RC1 |
| MindSpore | 2.6.0 |
| MindNLP | 0.4.1 |
| Python | 3.9 |

### 致谢
感谢MindSpore社区和昇腾AI开发者社区提供的技术支持和参考资料。
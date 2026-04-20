# ros2_robot_arm

[【开源实习】智能机械臂基于香橙派和 MindSpore 的深度学习 AI 推理功能和机械臂二次开发](https://atomgit.com/mindspore/community/issues/1770)

## 项目简介

本项目基于 OrangePi AI Pro、ROS2 与 YOLO 模型，实现积木识别、分拣和堆叠。

- 视觉推理：YOLO 检测积木位置
- 机械控制：机械臂抓取、分类、堆叠
- 可选模型路线：官方模型直接部署 / MindYOLO 自训练后转换 OM

## 环境要求

| 项目 | 说明 |
|---|---|
| 系统 | Ubuntu（开发板） |
| ROS | ROS2 Humble |
| Python | 3.9+ |
| 硬件 | OrangePi AI Pro、机械臂、USB 摄像头 |

## 总体预览

1. 硬件组装与连线。
2. 下载工程并安装依赖（ROS2、ais_bench、项目依赖）。
3. 模型准备（二选一）：
   - 方案 A：使用官方模型，放入两个 model 目录。
   - 方案 B：在云侧进行 MindYOLO 训练，导出 ONNX 并转换 OM，再部署到开发板。
4. 编译 ROS2 工作空间并加载环境。
5. 首次执行相机校准。
6. 运行单积木抓取，稳定后运行堆叠。

> 详细步骤、截图和完整命令请查看教程 notebook：
>
> [ros2_robot_arm_tutorial.ipynb](./ros2_robot_arm_tutorial.ipynb)

## 相关链接

- Issue: https://atomgit.com/mindspore/community/issues/1770
- MindYOLO 自训练参考: https://gitcode.com/shenyidong2025/mindyolo-cube.git

## 致谢

感谢 MindSpore 社区和昇腾 AI 开发者社区提供的支持。
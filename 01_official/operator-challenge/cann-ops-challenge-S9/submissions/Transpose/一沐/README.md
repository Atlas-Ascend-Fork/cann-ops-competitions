# Transpose — 一沐（v9）

## 版本与接口

本目录提交算子挑战赛 S9 的 Transpose v9 源码，团队名称为“一沐”。
实现按 dims 重排输入张量，语义对应 torch.permute；支持 float32、float16、int32、int8，输出为 ND。
Host、Kernel 及辅助头文件与 2026-09-15 的 v9 发布源码逐字节一致，本次只补齐源代码仓库需要的构建工程和说明。

## 目录

- build.sh、CMakeLists.txt、CMakePresets.json：构建入口和配置。
- op_host/：算子注册、形状推导与 tiling。
- op_kernel/：设备入口、分块流水线、连续复制和共享 tiling 结构。
- cmake/、scripts/：原 CANN 8.5 msopgen 工程的编译与安装支持文件，包含其原始版权声明及 makeself 许可证。
- Transpose.json：msopgen 算子描述。

构建引用 cmake/ 和 scripts/，这两项不是编译产物，不能省略。目录中不提交 .run、ZIP、缓存或性能日志。

## 构建

环境：Linux、Ascend 910B、CANN 8.5.0 SDK、CMake >= 3.19、Python 3、兼容的 GCC/G++。
请按 CANN 安装说明补齐 Python 编译依赖（例如 NumPy）；只有 SDK 文件但缺少 Python 依赖的基础镜像不能完成设备编译。
使用与目标运行环境相容的 CPU 架构和工具链进行原生编译。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd <本目录>
# SDK 路径不同时按实际位置设置：
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-8.5.0
bash build.sh
```

默认输出 build_out/custom_opp_*.run；默认并行度为 8，可通过 BUILD_JOBS 设置。
可用 bash build.sh cust_optiling 单独构建 Host tiling 库，或 bash build.sh binary 构建设备二进制。
msopgen 的内核生成器需要空构建目录；重复构建时，脚本将旧 build_out 移至本目录内唯一的 build_out.previous.* 备份目录，不删除旧产物，也不会清理调用者的工作目录。

若运行端为 Ubuntu 18.04 / GLIBC 2.27，必须使用相容环境编译 Host 库；在较新系统直接构建不会自动降低 GLIBC 依赖。
历史 v9 评测包的 Host 库使用 Ubuntu 18 / GCC 7.5 构建，最高引用 GLIBC_2.17。
仓库提交的是源码工程，不包含该历史二进制包，也不修改服务器系统 GLIBC。

## 核心实现

按 shape、stride 和 dtype 选择连续复制、连续后缀搬运、二维硬件转置与多维批次合并。
v9 在保留原设备内核的基础上，将相邻输入批次合并扩展到 16/32/64 行的短行布局。
int8 的 16 行分支额外限制输出连续性；不能满足条件时使用原路径。尾块按实际有效范围读写。
不根据输入值、隐藏用例编号或计时结果分派，无实验环境变量入口。

## 历史验证与限制

2026-11-04：本提交工程在 CANN 8.5.1 的独立容器中通过 Host 库、四种 dtype 的设备内核编译及 CPack 打包，生成 custom_opp_ubuntu_aarch64.run。验证使用 Git 导出的脚本权限，未映射 NPU，因此本次没有重新运行设备正确性或性能测试。该验证生成的二进制不提交到仓库。

2026-09-15 的 v9 兼容发布包通过 1388 组位精确回归与 19 组性能探针。
这些是历史 v9 发布包的验证，不表示每个新环境重新编译后均已复测。
隐藏官方用例的 shape 未公布，本文不宣称 v9 官方总时长低于 1000 us。

源文件 SHA-256：

```text
op_host/transpose.cpp
a075757d57742221b41df55fea69dc55bb6edc4a2f174a8fcba4506a33e7f581
op_kernel/transpose.cpp
6287cedcb9760462a0794de67dc6859081d672759954fa346b5f889b5415371c
op_kernel/transpose_pipeline.h
e124e5ec4bbad2dbed2d3366efd4005c47ac626ab4ba7374e59ee34513fffce0
```

# Concat — BatchLions

## 作品信息

- 团队：BatchLions。
- 单位：杭州瑞玉霞发科学技术事务所（个人独资）。
- 赛题：算子挑战赛 S9，Concat。
- 版本：v45-tight，原始提交包 SHA-256 为 `c11963b8f939b559685624909f4cedaf89afa85689ac5998a79dfec2f84cab0a`。
- 最终榜单：第十名，五档合计 **348.33 µs**，64 次提交。数据取自最终赛事榜单截图，本次合仓未重新运行隐藏用例。

本实现沿指定维度拼接动态数量的输入张量，支持赛题要求的 float16、float32、int8、int32。Host 规范化负轴，将输入归一为外部行数与各输入行宽，并生成 Tiling、核数、分带与布局参数。空分片保留原输入下标；满足条件时将仅一路非空的窄行转为连续复制。

## 构建与安装

环境要求：Linux，CANN 社区版 8.5.0，Ascend 910B，CMake 3.16 及以上、Python 3、C/C++ 编译器。先加载实际安装目录的环境，再从任意目录调用构建脚本：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh
```

SDK 安装位置可通过 `ASCEND_CANN_PACKAGE_PATH` 或 `ASCEND_HOME_PATH` 指定。默认构建 Release、目标 `ascend910b`；并行度可通过 `JOBS` 指定。产物位于 `build_out/custom_opp_*.run`。将产物拷至目标环境后，按安装包提示安装，并设置自定义 OPP 路径。

`build.sh binary` 只构建算子二进制；`CMakePresets.json` 保存构建默认值。顶层 CMake、`cmake/`、`scripts/` 均随工程提供。无需 TensorFlow 插件。

## 实现与优化

Host 根据行数、行宽、输入路数、对齐与 UB 容量选择按行、按列、DMA 拼装、Gather 拼装、按输入分核或输出线性切分。核数由平台查询，容量与对齐计算在纯 C++ 布局头中统一维护。

Gather 路径按输入打包连续读取多行，构建字节偏移表，在 UB 中组装结果后写出。int8 路径使用 half 中转并转换回原类型。手工缓冲轮转通过对应的流水事件维护依赖。

非对齐的单带窄行可将多行密排成超行，以减少碎片写出。自动模式检查建表摊销和 tile 容量；多带或不满足门槛时采用原布局。v45 的单带预算去掉跨带输入余量，保留逐路与块间的对齐空间。覆盖全部输入且至少 64 路的计划复用批量元数据加载；小路数及部分输入带使用原读取路径。

## 验证

以下测试无需 CANN 或 NPU，仅验证 Concat 的地址映射、输出覆盖、布局容量和缓冲边界：

```bash
bash tests/run.sh
```

- Concat 来源映射与地址回放：215242 项。
- 布局、对齐与 Host/Kernel 布局一致性：32681 项。
- 独立累计实际暂存需求：18000 组随机配置中 8247 个可行布局通过，包括 int8 转换缓冲容量。

离线检查不替代设备精度与性能测试。原 v45 包已有 CANN 8.5 编译记录；本次构建状态及 CI 结果以 PR 验证记录为准。

## 合仓范围

此目录仅包含 Concat v45 的实现、直接依赖、构建支持和 Concat 专用测试。Host、Kernel 入口和 TilingData 与原 v45 一致。共享 Tiling 头已删除其他算子的函数及专用常量。`source_version.json` 记录来源哈希与整理范围。

自身实现的许可见 `LICENSE`；生成的 CANN 工程支持文件保留原版权声明，makeself 的许可另见 `cmake/util/makeself/COPYING`。

## 🫐 Infra-Techno
- AI 基础设施/后端相关知识库
- 包含了个人对官方文档/原始技术的初步解读
- **说明：包含 AI 辅助创作**

---

### 🍉 DeepSeek OpenInfra
#### [Expert Parallelism Load Balancer (EPLB)](deepseek-openinfra/decipher/EPLB_decipher.md)
- DeepSeek 团队针对混合专家模型（**MoE**）开发的负载均衡算法
- 旨在解决多GPU环境下专家负载不均问题

#### [DeepGEMM](deepseek-openinfra/decipher/DeepGEMM_decipher.md)
- 专为 **​NVIDIA Hopper** 架构 GPU​ 优化的矩阵计算库
- 支持 FP8 浮点格式的密集矩阵和混合专家（MoE）模型的分组矩阵运算
- 实现超越传统专家调优库的性能表现

#### [Fire-Flyer File System (3FS) 预览](deepseek-openinfra/decipher/3FS_overview_zh-CN.md)
- 专为AI训练与推理工作负载设计的高性能 **分布式文件系统**
- 解决大规模数据处理中的存储挑战

#### [3FS 技术详解](deepseek-openinfra/decipher/3FS_decipher.md)
- 3FS的核心设计围绕高性能分布式存储展开，面向AI训练与推理场景中的大规模随机读取需求
- 采用分层架构与 **RDMA** 网络优化实现高吞吐与低延迟

#### [3FS-HDFS-MinIO 技术对比](deepseek-openinfra/decipher/3FS_HDFS_MinIO.md)
- 对比了另外两种常见的分布式存储
- **HDFS**: Hadoop Distributed File System
- **MinIO**: MinIO Object Storage

#### OFFCIAL DOCS
- 官方文档见 `deepseek-openinfra/docs_offcial` 文件夹
- 配套代码见 `deepseek-openinfra/code_offcial/code` 文件夹

---

### 🍉 通用
#### [大模型的能力边界 (2025)](general/The_Capability_Boundaries_of_Large_Models.md)
- 2025年大模型的能力边界本质是“语言与数据的双重囚笼”
- 受限于显性知识表达、统计学习范式及输入信息的完整性

---

## Reference
- **Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning**
- [📄 Paper Link](https://dl.acm.org/doi/10.1109/SC41406.2024.00089)  
- [📄 Arxiv Paper Link](https://arxiv.org/abs/2408.14158)
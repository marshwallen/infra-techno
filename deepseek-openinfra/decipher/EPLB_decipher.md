## 🍒 EPLB 文字详解

---

### **1 EPLB核心功能**
EPLB是DeepSeek团队针对混合专家模型（MoE）开发的负载均衡算法，全称为 Expert Parallelism Load Balancer，旨在解决多GPU环境下专家负载不均问题。其核心目标是通过动态调整专家分布，最大化GPU利用率并减少跨节点通信开销。

### **2 技术原理**
1. **冗余专家策略**  
   - 对高负载专家进行复制，分散计算压力。例如，若某专家处理任务量过大，EPLB会创建其副本，并将副本分配到空闲GPU上。
   - 结合DeepSeek-V3的**Group-Limited Expert Routing**（组限制专家路由）技术，优先将同一组的专家部署在同一物理节点，减少跨节点数据传输。

2. **分层负载均衡**  
   - **适用场景**：当服务器节点数能整除专家组数时（如预填充阶段）。
   - **步骤**：
     1. 将专家组均匀分配到各节点，确保节点间负载平衡。
     2. 在节点内复制专家，优化GPU间任务分配。
     3. 通过启发式算法打包专家副本至GPU。

3. **全局负载均衡**  
   - **适用场景**：专家组分布复杂或大规模并行时（如解码阶段）。
   - **策略**：忽略专家组限制，全局复制专家并动态分配至GPU，适应动态负载变化。

### **3 应用场景**
- **预填充阶段（Prefilling）**：采用分层策略优化小规模并行，例如处理4K序列长度的输入时，通过微批次重叠计算与通信。
- **解码阶段（Decoding）**：切换至全局策略，支持大规模并行（如EP128配置），减少GPU闲置时间。
- **跨节点优化**：通过组限制路由减少节点间通信数据量，提升分布式训练效率。

### **4 性能优势**
- **训练效率提升**：在DeepSeek-V3技术报告中，EPLB显著减少GPU闲置时间，推理阶段通信数据量降低30%。
- **资源利用率优化**：通过动态负载均衡，GPU利用率接近100%，适用于千亿参数规模的MoE模型。

---

## 🍩 EPLB 代码解读
- **源代码仓库**: https://github.com/deepseek-ai/eplb
- **核心 Code**: `deepseek-openinfra/docs_offcial/code/EPLB.py`

### 1. `balanced_packing` 负载均衡分组函数
```python
def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 n 个加权对象分配到 m 个分组中，确保每个分组包含 n/m 个对象，且所有分组的权重尽可能均衡。
    用于专家组的节点/GPU 级别负载均衡。
    
    Parameters:
        weight: [X, n] 每个专家的负载统计值（如计算量预测值）
        num_packs: 分组数量（如节点数或 GPU 数）
    
    Returns: 
        pack_index: 每个专家所属的分组编号
        rank_in_pack: 专家在分组内的排序（用于后续分配）
    """
    # 参数校验：确保总专家数可被分组数整除
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0  # 必须均分
    
    # 简单情况：每个分组仅1个专家时直接返回索引
    if groups_per_pack == 1:
        return ...  # 直接生成顺序索引
    
    # 核心逻辑：按负载降序排序后动态填充分组
    indices = weight.float().sort(-1, descending=True).indices.cpu()  # 按负载降序排序
    for group in indices[i]:
        # 选择当前负载最小的分组进行填充
        pack = min(range(num_packs), key=pack_weights.__getitem__)  # 贪心算法
        pack_index[i, group] = pack  # 记录分组编号
        rank_in_pack[i, group] = pack_items[pack]  # 记录组内顺序
        pack_weights[pack] += weight[i, group]  # 更新分组总负载
```

**关联技术说明**：
- 用于分层负载均衡策略中的 **专家组打包到节点**
- 与 `rebalance_experts_hierarchical` 的分层策略配合，优先保证节点间负载均衡

### 2. `replicate_experts` 专家复制函数
```python
def replicate_experts(weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    通过复制高负载专家生成物理副本，最小化所有副本的最大负载。
    
    Parameters:
        weight: [X, num_log] 逻辑专家的负载统计值
        num_phy: 复制后的总物理专家数
    
    Returns:
        phy2log: 物理专家对应的逻辑专家ID
        rank: 物理专家的副本序号
        logcnt: 每个逻辑专家的副本数量
    """
    # 初始化物理专家映射（初始为1:1映射）
    phy2log = torch.arange(num_phy, ...)  # 初始为直接映射
    
    # 动态复制高负载专家
    for i in range(num_log, num_phy):
        # 选择当前负载/副本数比值最大的专家进行复制
        redundant_indices = (weight / logcnt).max(dim=-1).indices  # 动态选择复制目标
        phy2log[:, i] = redundant_indices  # 记录副本所属逻辑专家
        rank[:, i] = logcnt[..., redundant_indices]  # 记录副本序号
        logcnt[..., redundant_indices] += 1  # 更新副本计数
```

**关联技术说明**：
- 实现 **冗余专家策略** 的核心逻辑
- 与 `rebalance_experts` 的全局负载均衡策略配合使用

### 3. `rebalance_experts_hierarchical` 分层负载均衡
```python
def rebalance_experts_hierarchical(...):
    """
    分层负载均衡策略实现：
    1. 将专家分组打包到节点（减少跨节点通信）
    2. 在节点内创建冗余专家副本
    3. 将物理专家打包到 GPU
    """
    # Step 1: 按组打包到节点
    tokens_per_group = ...  # 计算每组的负载总量
    group_pack_index = balanced_packing(...)  # 调用分组函数
    
    # Step 2: 节点内创建冗余副本
    tokens_per_mlog = weight.gather(...)  # 收集节点内专家负载
    phy2mlog = replicate_experts(...)  # 调用复制函数
    
    # Step 3: 物理专家分配到 GPU
    tokens_per_phy = ...  # 计算物理专家负载
    pack_index = balanced_packing(...)  # 再次调用分组函数
```

**关键设计**：
- 通过三级分配（节点→副本→GPU）实现 **组限制路由优化**
- 与 `rebalance_experts` 的入口逻辑配合，优先保证节点间负载均衡

### 4. `rebalance_experts` 入口函数
```python
def rebalance_experts(...) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    负载均衡器入口函数：
    根据系统配置选择分层策略（组可整除时）或全局策略（其他情况）
    """
    if num_groups % num_nodes == 0:
        return rebalance_experts_hierarchical(...)  # 分层策略
    else:
        return replicate_experts(...)  # 全局策略
```

**策略选择逻辑**：
- 分层策略优先保证 **节点内专家组的本地性**（减少跨节点通信）
- 全局策略适用于解码阶段等需要大规模并行场景

### 二、关键技术关联
1. **混合专家模型 (MoE)**  
   代码服务于 MoE 模型的专家并行策略，每个专家是独立的子网络，通过门控机制动态激活。

2. **流水线并行 (DualPipe)**  
   与流水线并行技术协同使用时，需注意通信与计算的时序重叠，EPLB 负责专家维度的负载均衡。

3. **通信优化技术**  
   - 组限制路由：通过 `balanced_packing` 实现同组专家节点内分配
   - 零气泡优化：在 `replicate_experts` 中通过动态副本分配减少等待时间

4. **动态负载预测**  
   代码中 `weight` 参数需通过历史数据预测（如移动平均法），需结合业务场景实现。

### 三、典型调用示例
```python
# 输入：2层 MoE 模型，每层12个专家
weight = torch.tensor([[90, 132, ...], [20, 107, ...]]) 

# 参数设置
phy2log, log2phy, logcnt = rebalance_experts(
    weight, 
    num_replicas=16,  # 总物理专家数
    num_groups=4,     # 专家组数
    num_nodes=2,      # 节点数
    num_gpus=8        # GPU总数
)
```
**输出解读**：  
`phy2log` 展示物理专家到逻辑专家的映射关系，同一逻辑专家可能对应多个物理副本（如 ID 5 对应两个副本）。


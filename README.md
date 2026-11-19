# UltraRun

在 Apple Silicon 上运行 Qwen 官方发布的 [Qwen3.8-27B BF16](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0) 文本模型。直接读取原始 safetensors。模型前向、DeltaNet Metal kernel、缓存、贪心解码、投机验证和 HTTP 服务由本项目实现。MLX 提供张量运算、GPU 调度和全注意力算子。

## 安装与下载

在 ARM64 Python 3.13 下执行。下载空间约 56 GB。

```sh
uv venv --python /opt/homebrew/bin/python3 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download.py
```

如果 Hugging Face 大文件下载不稳定，可以执行 `.venv/bin/python scripts/download.py --source modelscope`。两个路径使用同一份官方 SHA-256 清单，分片支持断点续传。

## 运行

```sh
.venv/bin/python -m localllm.cli verify-kernel
.venv/bin/python -m localllm.cli verify-model
.venv/bin/python -m localllm.cli generate "请用三句话解释统一内存。"
.venv/bin/python -m localllm.server --port 8765 --profile optimized --draft-schedule frontier_prefetch
```

```sh
curl -fsS http://127.0.0.1:8765/healthz
curl -fsS http://127.0.0.1:8765/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"请用三句话解释统一内存。","max_tokens":128}'
```

服务串行处理文本请求并一次性返回生成结果和指标。性能部署的 `optimized` 配置与 `compileddelta10` 基准共用 BF16 模型和草拟参数，使用主模型的完整词表验证候选。用 `--profile reference` 启动参考路径。`/healthz` 返回当前配置。CLI 的 `--thinking` 或 HTTP 请求字段 `"thinking": true` 启用官方模板的思考模式。当前实现覆盖文本，图像与视频需要视觉编码器。

打开 [本地聊天网页](http://127.0.0.1:8765/)，可以流式查看 Markdown 排版、代码块和每轮性能数据。标记为 `html` 的代码块可展开独立预览，预览禁用脚本。网页和终端共用 `/chat` 的生成、会话缓存及流式解码。Markdown 解析器使用本地的 [markdown-it 15.0.3](https://github.com/markdown-it/markdown-it/releases/tag/15.0.3) 浏览器构建。

服务启动后，可以在终端体验流式多轮聊天。回复会边生成边显示。`/reset` 清空会话，`/quit` 退出。也可以多次传入 `--prompt` 顺序执行多轮。

```sh
.venv/bin/python -m scripts.chat
```

脚本通过本机 `/chat` 使用同一份 BF16 主模型，显示每轮缓存命中、首 token 耗时、解码速度、实际调度和验证轮数。最近一个会话缓存提示词与已提交的回复前缀，核对模板生成的 token 后复用主模型 KV、DeltaNet 状态和原生 MTP KV。前缀变化时回退到仍匹配的检查点，或全量预填充。支持该调度的优化服务可在短对话与长输入上使用 `frontier_prefetch`，只流式发送经主模型确认的 token。退出、`/reset`、切换会话和普通 `/generate` 请求会释放会话缓存。

M2 Ultra 上用 `--max-tokens 1024` 跑固定中文 4096/1024 请求，流式解码为 51.63 token/s，复现历史输出文本和 213 轮验证。紧接的一轮复用 5119/5140 个输入 token，首 token 为 0.45 秒。见 `runs/chat-mtp-stream-verified.log`。

在 M2 Ultra 的优化配置下，聊天模板展开后恰好为 4096 个输入 token 且 `max_tokens` 为 1024 时，可使用 `"frontier_prefetch"`。启动参数 `--draft-schedule` 决定省略调度字段的请求使用什么模式，省略启动参数时为 `"standard"`。请求的 `draft_schedule` 字段优先，其他输入输出长度自动走 `"standard"`。`/healthz` 的 `default_draft_schedule` 显示启动配置，生成响应的 `metrics.draft_schedule` 显示实际生效的模式。可用 `"frontier"` 选择最多 11 个节点的 MTP 概率树，或用 `"logical"` 对照相同逻辑位置注意力下的常规草拟调度。逻辑位置归约会改变 BF16 舍入，这些模式可能改变生成结果。需要原有批次归约时，请求可显式指定 `"standard"`。

`"draft_schedule": "frontier_prefetch"` 在概率树上复用上一轮完整目标词表投影的 BF16 分数，每轮提前读取最多 12 行原版草拟权重。临时词表最多 160 行，超过 12 轮未刷新会过期。主模型仍用完整词表决定每个输出 token。

预取模式将主模型全注意力层和 MTP 草拟层的提示词 K/V 保存在只读缓冲区。生成时只更新后缀，注意力按原顺序归约两段数据。响应的 `metrics.target_prompt_kv_layers` 记录主模型启用分段的层数，`metrics.draft_prompt_kv_tokens` 记录 MTP 复用的提示词位置数。

主模型全注意力层对 8、16 位置的分段 K/V 验证批复用完整编译图。一个 Metal kernel 写入本轮 K/V 和上一轮接受的旁支，图按有效长度读取后缀，回滚行数作为输入传入。其他短验证批复用 QKV 投影及注意力输出至 FFN 的两段编译图。

词表预取模式还会从当前请求的提示词和已确认 token 中查找一条短语。常规短语最多补五个 token。若历史续写与原生 MTP 主链的前四个 token 一致，可提前结束本轮 MTP 展开。八 token 后缀一致，或六 token 后缀至少有两次一致的历史记录时，只需比对两个原生 MTP token。整棵树最多 16 个位置，已经选出的分支继续参与验证。补入的 token 由主模型逐个验证。

启用编译 DeltaNet 的优化配置会为词表预取模式编译 8、11 和 16 位置的紧凑分叉验证。分叉位置作为输入传入，同一批宽度和分叉数可复用编译图。优化服务在内存余量允许时，于就绪前用小型合成缓冲预热这些图和 6、11 位置的线性验证图，随后释放缓冲。

词表预取模式会按草拟深度和置信度统计本请求中已经验证的前四名命中情况，用它调整 MTP 分支展开顺序。已提交路径上的主模型首选计为一个样本，已经算出的未提交分支首选计四分之一个样本。校准可保留少量原生分数偏低的候选参与验证。超过八个验证位置前，调度会估计额外位置能确认多少 token。原生草拟分数仍用于约束验证宽度和后续 MTP 展开，节点和分叉数沿用原上限。

在至少观察到 24 个已打分的验证末端后，如果真实后继频繁落在 MTP 第二至第四名，词表预取模式会用拟合的路径到达概率调整候选展开顺序。分支准入和验证宽度仍按原累计概率计算。`metrics.draft_path_reorder_rounds` 记录请求内启用这一调整的轮数。

在词表预取模式下，优化服务对提示词中至多含 128 个不同 8K–32K 词表 ID 的请求启用分段草拟词表。单条高概率草拟路径先评估 8K 词表、已保留行与请求附加行，首选概率低于 0.90 时在 GPU 补扫其余原版 BF16 行。其余草拟评分使用 32K 词表。额外请求内存预留为 128 MiB，余量不足时回到完整草拟词表。`metrics.draft_conditional_vocab_attempts` 和 `metrics.draft_conditional_vocab_complements` 分别记录分段次数和补扫次数。

同一份带聊天模板的代码续写请求在两个 HTTP 配置中均使用 4096 个输入和 1024 个输出 token。参考和优化配置实测解码为 11.32 和 32.47 token/s，HTTP 响应耗时 104.27 和 47.75 s，首 token 为 13.86 和 16.21 s。返回文本哈希有变化。优化配置峰值 55.46 GiB，系统可用内存最低 45%。切换后验证进程锁和超额内存请求拒绝。结果记录在 `runs/http-profile-ab.log`。

## 统一内存

服务和基准通过进程锁互斥，确保工作区只加载一份模型。专用基准运行期间释放本机服务的模型，测完再恢复服务。服务对超出工作集预算或系统内存压力过高的请求返回 HTTP 503。

默认 MLX 工作集预算为 80 GiB，空闲分配器缓存目标为 2 GiB。这两个值可以用 `--memory-gib` 和 `--cache-gib` 调整。工作集预算还会按设备容量和 Metal 建议值收紧。请求开始前按预计 KV 增长和临时缓冲校验 Metal 预算，并基于 macOS 内存采样保留至少 12% 余量。余量不足时先释放空闲缓存、重新采样，仍不足则拒绝请求。加载、prefill 和 decode 会继续检查预算，服务在请求结束后释放空闲缓存。基准 JSONL 记录整段、prefill 和 decode 的 MLX 活动内存峰值，请求结束时的活动量和缓存量，以及请求期间观察到的系统可用内存最低百分比。

概率树在准入时额外预留约 220 MiB，覆盖最多 10 份分叉 MTP K/V 快照。每次最多并行扩展三条草拟路径。

词表预取再预留 256 MiB，覆盖分数缓冲和 BF16 词表重排期间并存的新旧缓冲。每次重排前重新检查 Metal 工作集。该词表只在当前请求内保留。

## 性能基准

```sh
.venv/bin/python -m localllm.cli bench --runs 3 --jsonl runs/baseline.jsonl
.venv/bin/python scripts/tune.py
```

默认扫参运行基线和两候选 MTP。可在命令末尾指定候选配置。单个试验超时后会结束该进程并释放工作集。

独占 Metal 工作区后，可用 `.venv/bin/python scripts/tune.py logical10 frontier11` 对照两种调度。

`frontierprefetch11` 基准启用词表预取。

默认取当前 Python 标准库 `inspect.py` 的前 4096 个 Qwen token，做代码续写。指定 `--prompt-file` 可以换用至少 4096 token 的文本。每次预热后清空请求缓存，贪心生成 1024 个 token，基准在 EOS 后继续计数到 1024。结果记录输入与输出 token 的哈希、TTFT、prefill、峰值显存以及两种吞吐。

- `decode_tok_s` 统计首 token 之后的 1023 个已确认 token。30、50、100 token/s 三阶段按这项单请求指标验收。
- `end_to_end_tok_s` 统计从 4096-token prefill 开始到生成结束的 1024 个 token。
- `--merge` 拼接 BF16 投影矩阵以减少发射次数。`--skinny` 为多 token 验证的窄矩阵乘法选择更快的 MLX 路径。
- `--vector-bf16` 用自写 Metal kernel 验证 2–4 个 token。每次读取一组 BF16 权重并供该组 token 复用，分块 FP32 累加后写回 BF16。
- `--fused-delta` 在短块 DeltaNet 路径上融合四抽头因果卷积与 SiLU，并在递推 kernel 中归一化 Q/K。此选项保留用于命令行基准。
- `--compact-delta` 为短线性验证块记录 FP32 秩一更新。候选部分被拒绝时，用原始乘法和 FMA 顺序恢复已接受前缀的状态。复测命令为 `.venv/bin/python scripts/tune.py residual5 compactdelta5`。
- `--compiled-delta` 为 6 和 11 位置的线性 DeltaNet 验证缓存 MLX 计算图。复测命令为 `.venv/bin/python scripts/tune.py shareddraft10 compileddelta10`。
- `--wide-blocks` 将线性验证扩至 16 token，两组 SIMD 累加器共享 BF16 权重读取。配合 `--draft-extension-confidence 0.5`，设备端每草拟 5 个候选检查一次累计概率，达到阈值才延长。复测命令为 `.venv/bin/python scripts/tune.py compactdelta5 wideblock10 selective10`。
- `--draft-panel-checkpoint` 在第 7 个候选后复用延长阈值。另一个候选连同待验证 token 会进入第二个 SIMD 面板。复测命令为 `.venv/bin/python scripts/tune.py compacttree10 panelcheck10`。
- `--draft-root-branch-gap` 在设备草拟首位置的前两名分差足够小时，验证最多两个备选位置。主链使用因果注意力，备选链按自己的祖先掩码计算。复测命令为 `.venv/bin/python scripts/tune.py selective10 rootbranch10`。
- `--compact-tree` 在寄存器保留单分叉状态，记录每个节点的 FP32 更新量和主链末状态。接受时沿选中路径恢复状态。与 `--wide-blocks` 同用时，树宽上限由 8 增至 16。复测命令为 `.venv/bin/python scripts/tune.py rootbranch10 compacttree10`。
- `--draft-branch-positions 3` 在前三个草拟位置比较第二候选的累计概率，选择一个分支。复测命令为 `.venv/bin/python scripts/tune.py compacttree10 adaptivebranch10`。
- `--draft-branch-depth 5` 从分叉点的 MTP 缓存继续草拟备选链。实际长度由剩余输出、验证树的节点预算和该分支的续写置信度约束。复测命令为 `.venv/bin/python scripts/tune.py compacttree10 longbranch10`。
- `--draft-second-branch` 在第三名也满足分差阈值时，为同一个分叉点增加一个验证节点。紧凑树上的两条备选链共用一个 FP32 分叉状态。复测命令为 `.venv/bin/python scripts/tune.py compacttree10 secondbranch10`。
- `--draft-multi-fork` 在同一紧凑验证树中为主链上最多两个不同位置添加备选链。两个分叉状态保存在 DeltaNet kernel 的寄存器中，接受时复用秩一回滚。复测命令为 `.venv/bin/python scripts/tune.py adaptivebranch10 multifork10`。
- `--wide-verify` 在 5-token 验证批次上为两个已测形状选择向量化 BF16 投影，并分块评分完整词表。可运行 `.venv/bin/python scripts/tune.py fused4 wide4` 对比。
- `--packed-ffn` 按 SIMD 的读取顺序逐位重排合并后的 FFN 输入矩阵，供 5 或 7 token 批次使用。该实验路径为 prefill 保留原矩阵，加载时依据剩余内存逐层启用。对照命令为 `.venv/bin/python scripts/tune.py wide4 packed4`。
- `--packed-main` 用同一种逐位可逆的排布替换常驻的 FFN 输入、FFN 输出和 DeltaNet 输入矩阵。短块直接读取重排后的 BF16，prefill 按四层分段临时解包。复测命令为 `.venv/bin/python scripts/tune.py packed4 mainpacked4 mainpacked6`。
- `--layer-prefill` 在最多 4096 个输入 token 的窗口内每层解包一次权重。将完整的 512-token 块合批执行 FFN，余块单独处理。M2 Ultra 优化配置在首个完整 4K 窗口分两组执行 DeltaNet，组内小型 `a/b` 投影沿用每 512 行的归约。可用 `.venv/bin/python scripts/tune.py compileddelta10 layerprefill10` 对照。
- `--lossless-down` 将 64 个 FFN down 矩阵按四个片段编码，指数范围超出的片段包保留原 BF16。加载器逐位验证全部矩阵。验证 kernel 在寄存器中还原 BF16，prefill 按层解为行优先。优化配置启用此选项，原始 checkpoint 可直接复用。
- `--paired-down` 让两个 FFN down 输出块共用激活片段，保留原来的 512 列 FP32 分片和归约顺序。可用 `.venv/bin/python scripts/tune.py losslessvocab10 paireddown10` 对照。
- `--lossless-gate-up` 联合编码 FFN 的 gate/up 权重。验证 kernel 复用两路激活片段，在寄存器中还原 BF16，再在同一线程组完成 SwiGLU。优化配置启用此选项，加载器逐位回读全部 64 个矩阵。
- `--lossless-inputs` 将 48 个 DeltaNet 输入和 16 个全注意力输入矩阵按相同方式无损编码。短验证块在同一线程组内按原有八组 FP32 累加顺序合并，prefill 按层恢复行优先。优化配置启用此选项。
- `--lossless-attention-outputs` 为 64 个全注意力与 DeltaNet 输出矩阵启用同样的编码，可用 `.venv/bin/python scripts/tune.py compacttree10 losslessoutputs10` 对照内存和吞吐。
- `--packed-head` 为主模型的短块验证使用逐位重排的输出投影，始终给完整词表评分。M2 Ultra 配合 MLX 0.32.4 时，Qwen3.8-27B 的完整词表投影在一个线程组内合并 FP32 分片并选出局部 BF16 最大值。优化配置同时启用 `--packed-head-only`，常驻一份输出权重，首 token 按块解包，草拟直接取候选行。与 `--draft-vocab 32768` 同用时，仅草拟限定为前 32768 个词表 ID 和输入中出现的 ID。
- `--lossless-vocabulary` 以分段可逆 BF16 编码保存完整输出权重，加载时逐段校验。短块验证直接在 Metal kernel 内读取编码后的权重，首 token 和草拟按需恢复词表行。可用 `.venv/bin/python scripts/tune.py compacttree10 losslessvocab10` 对照。
- `--packed-attention` 在 `--packed-main` 基础上覆盖注意力 Q/K/V、注意力输出和 DeltaNet 输出，共替换 256 个矩阵。复测命令为 `.venv/bin/python scripts/tune.py packedhead4 packedattn4 packedattn5`。
- `--fast-draft` 为 MTP 更新所有 K/V，仅计算末位置的查询和 FFN 输出。`--fused-ffn` 将分块求和、SwiGLU 和下一次投影所需的重排合并。M2 Ultra 上，Qwen3.8-27B 的 9–16 位置验证将 gate/up 投影与 SwiGLU 放在同一线程组中，并复用输入片段。
- `--fused-delta-io` 在短线性验证块中融合 DeltaNet 的投影求和、卷积及可回滚历史，归一化和门控直接写入下一投影的排布。复测命令为 `.venv/bin/python scripts/tune.py fusedffn4 deltaio4`。
- `--split-attention` 将有 KV 缓存的 6–8 token 线性验证查询均分为两段，每段按各自末位置读取同一份 K/V 的因果前缀。复测命令为 `.venv/bin/python scripts/tune.py fusedffn5 splitattn5`。
- `--draft-confidence` 用草拟词表内的累计概率决定何时停止延长候选链，已草拟 token 仍交给主模型验证。试验最多草拟五个，低于阈值时至少保留两个。复测命令为 `.venv/bin/python scripts/tune.py splitattn5 adaptive5 compact5`。
- `--device-draft` 在 GPU 上串起线性 MTP 草拟并一次读回。置信度规则在读回后决定提交多少候选，因此 GPU 可能计算未提交的末端候选。复测命令为 `.venv/bin/python scripts/tune.py adaptive5 device5`。
- `--fused-draft-ranking` 在 Metal 上同时计算草拟词表的 argmax 和归一化置信度。复测命令为 `.venv/bin/python scripts/tune.py compacttree10 draftfused10`。
- `--cache-draft-embedding` 分块预计算 MTP 输入投影中只依赖 embedding 的 FP32 部分。提示词额外条目使用独立缓冲。复测命令为 `.venv/bin/python scripts/tune.py compacttree10 cachedembed10`。
- `--draft-lane-cache` 缓存原 MTP 单 token 输入投影中独立于隐藏状态的 FP32 累加值，并保留 MLX 的累加与归约顺序。在 M2 Ultra、MLX 0.32.4 的设备端草拟路径上，对至少 4096 输入、生成预算至少 1024 的请求启用。每个请求最多缓存 128 个输入 token，表与索引共约 81 MiB，内存余量不足时跳过建表。
- `--draft-paired-root` 在上述长请求的根分支上合批计算主、备续写，使用原 BF16 的单行投影累加顺序。每个分支各自更新 K/V。按原阈值选择根分支时，直接复用该备选续写。
- `--fused-residual` 将短块的注意力残差加法、FFN 前 RMSNorm 和 BF16 排布重排合为一次 Metal 调用。复测命令为 `.venv/bin/python scripts/tune.py device5 residual5`。
- `--reserve-kv` 为多 token 路径和 MTP 单 token 路径按 512 token 预留 KV 容量。主模型的单 token 路径使用追加缓存。
- `--draft-depth N` 使用原始权重中的 MTP 层。`--draft-vocab N` 可缩小 MTP 候选词表。`--lookup-depth N` 可从请求上下文中生成候选。所有候选经过完整 BF16 主模型和完整词表验证。性能扫参同时检查输出 token 哈希。
- `--lossless-draft-vocab` 在加载时无损编码草拟词表的固定部分。每个请求仅临时读取提示词额外出现的行。`lossless_draft_vocab_rows` 记录加载的编码行数。复测可运行 `.venv/bin/python scripts/tune.py losslessvocab10 losslessdraft10`。
- `--shared-vocabulary` 让主模型验证和固定草拟词表共用一份 BF16 编码。词表取行和并列打分按原 token ID 处理。可用 `.venv/bin/python scripts/tune.py losslessdraft10 shareddraft10` 对照。
- 树形验证支持一次前向中的主链和备选分支。常规树最多 8 节点，宽块紧凑树最多 16 节点。每个节点只读取其祖先，接受时只提交选中路径的注意力缓存和 DeltaNet 状态。`--tree-depth` 增加备选续写，`--tree-tail-gap` 根据倒数第二个候选的分数差在固定预算内分支。两者用于命令行实验。

### 首轮 M2 Ultra 实测

| 4096 输入 / 1024 输出 | 解码 token/s | 端到端 token/s | 首 token |
| --- | ---: | ---: | ---: |
| 逐 token 贪心基线 | 11.33 | 9.80 | 14.18 s |
| MTP 两候选，单份模型驻留复测 | 16.88 | 13.84 | 13.39 s |
| MTP 两候选，短块 DeltaNet 融合 | 17.28 | 14.11 | 13.36 s |
| MTP 四候选，短块 DeltaNet 融合 | 19.19 | 15.36 | 13.35 s |
| MTP 四候选，五 token 专用投影 | 20.04 | 15.88 | 13.43 s |
| 五 token 路径，与七节点树同场复测 | 20.43 | 16.15 | 13.30 s |
| 七节点树 | 18.21 | 14.71 | 13.42 s |
| 五 token 路径，与固定预算树同场复测 | 19.04 | 15.27 | 13.34 s |
| 固定预算树 | 16.91 | 13.53 | 15.19 s |
| 五 token 路径，与 FFN 重排同场复测 | 20.48 | 16.17 | 13.40 s |
| 五 token 路径，62 层 BF16 FFN 重排 | 21.95 | 17.09 | 13.32 s |
| 双排布，同场复测 | 22.02 | 17.13 | 13.31 s |
| 单排布，MTP 四候选 | 24.79 | 17.81 | 16.24 s |
| 单排布，MTP 六候选 | 21.58 | 16.07 | 16.32 s |
| 单排布，四候选同场对照 | 24.98 | 17.89 | 16.27 s |
| 单排布，缩小草拟候选词表 | 26.35 | 18.52 | 16.47 s |
| 缩小草拟候选词表并重排完整输出投影 | 26.76 | 18.73 | 16.43 s |
| 同场对照，原注意力投影 | 26.11 | 18.46 | 16.29 s |
| 重排注意力投影，四候选 | 27.29 | 18.99 | 16.43 s |
| 重排注意力投影，五候选 | 21.91 | 16.01 | 17.29 s |
| 草拟与 FFN 融合同场对照 | 27.58 | 19.16 | 16.36 s |
| MTP 只计算末位置的输出 | 27.68 | 19.25 | 16.24 s |
| MTP 末位置输出和 FFN 融合 | 27.79 | 19.27 | 16.32 s |
| DeltaNet 输入输出融合的同场对照 | 28.36 | 19.58 | 16.22 s |
| DeltaNet 输入输出融合 | 27.35 | 19.12 | 16.16 s |
| 五候选，原注意力路径 | 26.05 | 18.47 | 16.16 s |
| 五候选，分段因果注意力 | 29.18 | 19.98 | 16.20 s |
| 分段注意力，与缓冲批次同场复测 | 28.63 | 19.71 | 16.24 s |
| 分段注意力，增大 Metal 命令缓冲批次 | 28.89 | 18.93 | 18.69 s |
| 草拟策略同场对照 | 28.84 | 19.83 | 16.18 s |
| 草拟概率截断 | 29.22 | 19.98 | 16.25 s |
| 草拟词表缩至前 8192 个词表 ID 加输入 ID | 23.83 | 16.95 | 17.47 s |
| 置信度截断，五候选复测 | 29.23 | 19.98 | 16.27 s |
| 置信度截断，六候选 | 29.06 | 19.93 | 16.18 s |
| GPU 整链草拟同场对照 | 29.01 | 19.91 | 16.17 s |
| GPU 整链草拟 | 29.53 | 20.14 | 16.19 s |
| 残差融合的同场对照 | 29.52 | 20.15 | 16.15 s |
| GPU 整链草拟与残差融合 | 30.28 | 20.47 | 16.24 s |

复测将主模型解码前向从 1023 次降至 417 次，MLX 峰值 52.38 GiB，系统可用内存最低 45%。MTP 输出 token 哈希与逐 token 基线有差异，服务默认使用逐 token 路径。完整结果见 `runs/tuning.jsonl`。新结果记录推理源码哈希，便于区分使用同一参数的不同构建。

短块融合与同场 MTP 对照分别为 17.28 和 16.89 token/s。对照中主模型验证占解码时间约 92%。融合版通过 kernel 数值校验，整段生成 token 哈希有变化。两组的 MLX 峰值均为 52.38 GiB。

四候选融合路径与两候选融合路径在该次 4K/1K 输入上生成了相同的 token 序列。五 token 专用投影将四候选路径的平均主模型验证耗时从 137.69 ms 降至 124.43 ms，输出 token 哈希有变化。两组的 MLX 峰值均为 52.38 GiB，系统可用内存最低 45%。

七节点树把主模型验证从 344 轮降至 316 轮，备选分支命中 36 轮。七节点和固定五节点树的每轮验证开销均高于各自对照。七节点树峰值 52.44 GiB，固定预算树峰值 52.38 GiB。整模校验覆盖分支隔离、备选路径提交和下一 token，详见 `runs/tree-verify-ab.log` 与 `runs/tailtree-ab.log`。

FFN 重排将平均单轮主模型验证由 122.48 ms 降至 103.87 ms。权重逐位回读一致，整段输出哈希与对照有变化。实验路径保留两种 FFN 排布，MLX 峰值为 72.97 GiB，系统可用内存最低 40%。真实形状微基准和完整对照分别在 `runs/packed-mma-probe.log` 和 `runs/packed-4k-ab.log`。

单排布替换了 176 个矩阵。相比同场双排布，模型常驻内存由 71.48 GiB 降到 50.89 GiB，MLX 峰值由 72.97 GiB 降到 52.98 GiB。四候选版系统可用内存最低 45%。六候选版减少了主模型前向轮数，草拟和每轮验证耗时均上升。整模结果在 `runs/packed-main-ab.log`。

候选词表与输出投影三组对照生成了逐 token 一致的 1024 个输出。草拟耗时从 7.29 s 降到 3.58 s。输出投影原始 BF16 的位模式经全矩阵回读校验，组合配置峰值为 55.35 GiB，系统可用内存最低 44%。见 `runs/packed-head-ab.log`。

新增 80 个注意力投影矩阵的重排将平均验证耗时从 103.28 ms 降至 96.39 ms。四候选版 MLX 峰值为 55.47 GiB，系统可用内存最低 44%，输出 token 哈希有变化。五候选版平均验证耗时为 132.43 ms。见 `runs/packed-attention-ab.log`。逐位可逆指数编码的微基准低于直接读取重排 BF16 的吞吐，保留为实验记录 `runs/packed-exponent-probe.log`。

只计算末位置后，MTP 的 K/V 逐位一致，4K/1K 输出哈希与同场对照一致。融合 FFN 通过数值校验，整段输出哈希有变化，平均每轮主模型验证未见提速。组合版峰值 55.46 GiB，系统可用内存最低 44%。见 `runs/fused-ffn-ab.log`。

DeltaNet 输入输出融合通过卷积历史和门控数值校验，平均单轮验证从 94.21 ms 降至 93.53 ms。输出哈希有变化，验证轮数由 343 增至 358，两组的 MLX 峰值均为 55.46 GiB。见 `runs/delta-io-ab.log`。

交错批宽测试用同一份 4K KV 缓存得到 5 token 验证 93.46 ms、6 token 验证 103.63 ms 的中位耗时。分段注意力使完整五候选基准的单轮验证从 107.41 ms 降到 96.56 ms，1024 个输出 token 与四候选融合路径一致。两组峰值 55.46 GiB，系统可用内存最低 44%。见 `runs/target-width-probe.log` 与 `runs/split-attention-ab.log`。

进程级命令缓冲对照 `.venv/bin/python scripts/tune.py splitattn5 buffered5` 中，较大批次生成相同 token，首 token 延迟增至 18.69 s，MLX 峰值增至 57.37 GiB。服务和默认配置保留 MLX 的原缓冲设置。见 `runs/metal-buffer-ab.log`。

草拟概率截断在 41 轮提前结束，少草拟 62 个候选，仍确认 704 个候选。三组输出 token 哈希一致。8192 词表组多跑 12 轮，解码更慢。直接使用 BF16 操作数的 SIMD 微基准与现有精确扩宽路径数值一致，未见稳定吞吐改善。见 `runs/adaptive-draft-ab.log` 与 `runs/native-bf16-mma-probe.log`。

6 token 共享 K/V 的自写注意力与现有分段路径的微基准接近，分别为 0.453 ms 和 0.454 ms。六候选减少六轮验证，新增草拟抵消了收益。GPU 整链草拟和同场对照的 1024 个输出 token、验证轮数和候选接受数均一致，草拟耗时从 3.96 s 降到 3.54 s，峰值为 55.46 GiB。见 `runs/short-sdpa-merge-probe.log`、`runs/adaptive-depth-ab.log` 与 `runs/device-draft-ab.log`。

残差融合逐位保留 BF16 残差加法的结果，RMSNorm 通过容差校验。4K/1K 测试的输出 token 哈希有变化，主模型验证从 319 轮降到 311 轮，平均单轮验证为 96.34 ms 和 96.64 ms。两组峰值均为 55.46 GiB，融合组系统可用内存最低 43%。同一输入复测得到 30.36 和 29.63 token/s，三轮的中位数为 30.28 token/s，输出 token 哈希一致。见 `runs/residual-fusion-ab.log` 和 `runs/residual-fusion-repeat.log`。

DeltaNet 秩一回滚通过 1–8 个 token 的状态逐位校验。两次 4K/1K 对照均生成相同的 1024 个 token，并执行 311 轮主模型验证。初测分别为 30.37 和 30.39 token/s，复测分别为 30.29 和 29.86 token/s。decode 活动内存峰值从 55.03 GiB 降到 54.35 GiB，整段峰值出现在 prefill，为 55.46 GiB。见 `runs/compact-delta-ab.log` 和 `runs/compact-delta-memory-ab.log`。

将回滚内联到下一轮递推后，状态通过逐位校验，整段输出也与同场两组对照一致。完整快照、独立恢复和内联恢复分别为 30.50、30.84 和 29.65 token/s。两种恢复方式解码峰值均为 54.35 GiB，内联的主模型验证累计耗时从 29.53 s 增至 30.79 s。实验实现已移除，保留独立恢复配置和 `runs/inline-delta-ab.log`。

FFN 微基准对比同一线程组内的片段交错和每个 lane 的四片段向量读取。两种布局的 BF16 权重回读和 FP32 分块累加均逐位一致，加载两层真实权重测试时峰值为 3.74 GiB。5 token 的 FFN 输入投影从 477.49 GB/s 提升到 497.95 和 501.05 GB/s。交错布局的整模解码为 30.65 token/s，原布局同场为 30.69 token/s。输出序列一致，两组解码峰值均为 54.35 GiB。已撤回交错布局的整模接入。见 `runs/ffn-streaming-probe.log` 和 `runs/interleaved-ffn-ab.log`。

双组累加的 BF16 投影和 FFN 在 9、13、16 token 上与分组独立计算逐位一致。十候选配置的 4K/1K 首轮为 29.42 token/s，同场五候选为 30.25 token/s。提高第五个候选后的延长阈值后，复测为 31.16 token/s，同场五候选为 30.46 token/s。主模型验证从 311 轮降到 286 轮，其中 109 轮验证至少 9 个位置。所有输出 token 一致。解码峰值从 54.35 GiB 增至 54.47 GiB，整段峰值 55.46 GiB。见 `runs/wide-block-ab.log` 和 `runs/wide-selective-ab.log`。

11 个查询、4107 个 KV 位置时，三个查询共享读取的微基准为 0.583 ms，分段注意力为 0.631 ms。跨查询头共享和分组矩阵乘法在此宽度下耗时更高。共享读取接入整模后，平均单轮主模型验证从 98.88 ms 降到 98.06 ms。输出哈希有变化，主模型验证从 286 轮增到 331 轮，解码从 30.96 降到 27.07 token/s。实验路径已撤回。见 `runs/wide-attention-probe.log`、`runs/wide-attention-cross-probe.log` 和 `runs/wide-shared-attention-ab.log`。

从 safetensors 偏移量合计得到 51.25 GB 的主模型层和完整词表投影权重。按每轮读取一次、[M2 Ultra 的标称 800 GB/s](https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/) 计算，仅这部分理想读取时间约为 64 ms。31.16 token/s 的运行平均每轮确认 3.58 个 token，耗时 114.81 ms。按当前轮耗时，50 token/s 需要每轮平均确认约 5.74 个 token。实际所需确认数随验证轮耗时变化。

对 `selective10` 的输出保持一致的排名探针记录了 261 次首处拒绝。92 次的正确 token 排在第二，14 次位于草拟词表外。首位置 69 次拒绝中有 32 次排在第二。首位置分差阈值设为 1.0 后，整模试验在 38 轮验证备选链，命中 10 轮并确认其上的 16 个候选。验证轮数从 286 降到 278，吞吐为 31.00 token/s，同场对照为 30.87 token/s。整段输出哈希有变化，解码峰值由 54.47 GiB 增至 55.34 GiB。见 `runs/draft-rank-probe.log` 和 `runs/root-branch-ab.log`。

单分叉秩一回滚通过了 16 节点、三个分叉位置和所有前缀状态的逐位校验。4K/1K 实测树宽最高 13，解码峰值从 55.34 GiB 降至 54.51 GiB。两组均运行 278 轮，输出 token 逐个一致，吞吐分别为 31.21 和 31.09 token/s。整段峰值为 55.46 GiB，系统可用内存最低 44%。见 `runs/compact-tree-retest.log`。

在前三个位置选择分支时，分别尝试 53、48 和 36 次，命中 15、6 和 5 次。首位置固定分支在同场对照命中 10 次。自适应分支增加了确认的分支候选，由 16 增至 41，但输出哈希有变化，主模型验证由 278 轮增至 287 轮。4K/1K 解码分别为 31.28 和 29.96 token/s，解码峰值为 54.53 和 54.55 GiB。见 `runs/adaptive-branch-ab.log`。

在第 5 和第 10 个候选后分别检查置信度，最多草拟 15 个候选。完整 4K/1K 生成中有 3 轮接受了全部 15 个候选。相比同场十候选配置，验证轮数从 278 降到 275，草拟时间从 4.57 s 增至 4.98 s。两组 1024 个输出 token 一致，吞吐分别为 31.40 和 30.73 token/s，整段峰值均为 55.46 GiB。见 `runs/staged-draft-ab.log`。

实际 4K/1K 验证路径的计算图构建平均耗时 4.67 ms。每七轮同步采样一次，共 40 轮，64 层前向的中位耗时为 91.52 ms，完整词表投影为 3.98 ms。验证位置至多 8 个和超过 8 个时，层栈采样中位数分别为 87.39 和 96.00 ms。采样运行的 1024 个输出 token 与十候选对照一致。见 `runs/verification-cost-probe.log`。

BF16 无损微基准将符号和尾数与指数差分开存放，超范围瓦片按原样放入稀疏缓冲。逐瓦片指数基准和整矩阵共用基准均逐位还原了两层真实权重和 FP32 投影部分和。编码占原始大小约 78–79%。共用基准的 6 token FFN 投影接近直接读取，11 token 的输入和输出分别由 0.8101 ms、0.3593 ms 增到 0.9643 ms、0.4181 ms。两个微基准的 MLX 峰值分别为 3.53、3.78 GiB。见 `runs/bf16-planes-probe.log` 与 `runs/bf16-fixed-probe.log`。

上下文候选探针在 4K/1K 有效草拟前缀上，从已提交历史检索长度为 4、6、8 或 12 的最长后缀，并要求该后缀只有一种已观察续写。检索与 MTP 第一名有分歧时，直接采用检索会修正 8 次错失并破坏 25 次正确草拟。限定检索结果为 MTP 第二名时为 7 次修正、10 次破坏。探针保留原草拟策略，输出哈希与 `selective10` 对照一致。见 `runs/lookup-rerank-probe.log`。

在两层真实 FFN 输入矩阵上，Metal 只读测得 609.75 GB/s。优先调度相邻 K 分片时，6 token 输入投影从 498.71 提升到 541.62 GB/s，FP32 部分和逐位一致。串联八层 FFN 后，6 token 耗时为 6.5361 和 6.5315 ms，11 token 为 6.8154 和 6.7782 ms。整模 4K/1K 的 278 轮验证和输出逐个一致，解码分别为 31.17 和 31.12 token/s。已撤回整模调度选项。见 `runs/packed-dispatch-probe.log`、`runs/ffn-chain-probe.log` 和 `runs/k-order-ffn-ab.log`。

融合注意力前 RMSNorm、BF16 写回和片段重排，在 5120 维的测试形状上逐位匹配原算子。4K/1K 整模两组输出逐个一致，验证均为 278 轮，解码分别为 31.31 和 31.27 token/s。真实 FFN 输出投影的另一项实验同时融合分块求和、残差和下一层归一化，11 token 耗时由 0.3940 ms 增至 0.4186 ms，测试中有少量 BF16 位差。运行时保留现有路径。见 `runs/input-norm-ab.log` 和 `runs/ffn-epilogue-probe.log`。

草拟归约的独立耗时从 0.13111 ms 降到 0.11109 ms，包含真实 BF16 词表投影时为 0.56357 和 0.56325 ms。两次交换顺序的 4K/1K 对照中，融合分别节省了 0.059 和 0.076 s 草拟时间，均通过 278 轮验证并生成相同的 1024 个 token。两组峰值均为 55.46 GiB。见 `runs/draft-ranking-ab.log` 和 `runs/draft-ranking-retest.log`。

相同的 4K/1K HTTP 请求使用融合和恢复原归约的配置分别实测 31.84 和 31.53 token/s，生成文本相同，均执行 266 轮验证。此前同一请求的优化服务记录为 32.47 token/s。融合归约保留为命令行实验选项，在线配置恢复为 `compacttree10`。见 `runs/http-draft-ranking.log` 和 `runs/http-draft-ranking-restored.log`。

拒绝尾部探针检查上轮主模型在错误草拟分支上的预测，能否作为下一轮候选。261 次有剩余预测的轮次中，206 次与新一轮 MTP 首候选不同。不同的首候选仅 4 次匹配真实续写，若直接替换会错过 154 次正确的 MTP 首候选。该探针收集数据，保留原解码路径和输出序列。见 `runs/rejected-tail-probe.log`。

第七候选后的检查点将至少 9 个验证位置的轮数从 120 降到 91，草拟打分从 2055 降到 1950，主模型验证由 278 增至 279 轮。两次交换顺序的 4K/1K 对照输出一致，草拟耗时分别减少 0.145 和 0.133 秒。解码分别为 31.40 对 31.19，以及 31.34 对 31.31 token/s。两组整段峰值 55.46 GiB，系统可用内存最低 44%。见 `runs/panel-checkpoint-ab.log` 和 `runs/panel-checkpoint-retest.log`。

五候选备选链在 4K/1K 实测中用到 16 个验证位置。有 3 轮接受了完整的五候选备选链。相较两候选备选链，分支确认数从 16 增至 31，主模型验证从 278 降至 276 轮，草拟时间从 4.58 增至 4.82 s。输出哈希有变化，解码为 31.19 token/s，同场对照为 31.23。解码峰值为 54.60 GiB，整段峰值为 55.46 GiB。见 `runs/long-branch-ab.log`。

同一分叉点的多备选链通过 16 节点逐位校验。第三名在 4K/1K 中加入了 12 轮，当次均未命中。两组执行 278 轮，输出逐个一致，解码为 31.45 和 31.33 token/s。解码峰值均为 54.53 GiB，整段峰值 55.46 GiB，系统可用内存最低 44%。见 `runs/second-branch-ab.log`。

真实 MTP 权重微基准中，单 token 输入投影含归一化时从 0.170 ms 降到 0.078 ms。拆分 FP32 累加改变约 0.037% 的 BF16 投影结果。两次 4K/1K 对照的输出 token 一致，各节省约 0.18 s 草拟时间，验证由 278 增至 279 轮。独立缓冲复测为 31.38 对 31.19 token/s。拆分缓冲使实验峰值从 56.86 GiB 降到 56.24 GiB，当场基线为 55.46 GiB。系统可用内存最低 44%。见 `runs/mtp-embedding-probe.log`、`runs/cached-mtp-embedding-ab.log` 和 `runs/cached-mtp-embedding-retest.log`。

将 MTP 草拟注意力和 KV 缓存限制到最近 512 个位置后，4K/1K 草拟耗时从 4.62 降至 4.43 s，解码峰值从 54.53 降至 54.50 GiB。主模型仍用完整上下文验证候选，验证轮数从 278 增至 324，解码从 31.13 降至 28.72 token/s，输出 token 哈希有变化。两组系统可用内存最低均为 44%。已撤掉滑动窗口实现，服务使用全长 MTP KV。见 `runs/draft-window-retest.log`。

两层真实权重的微基准按 K 分片连续排布 FFN 与 DeltaNet 输入矩阵。重排往返和 6、11、13 个验证位置的 FP32 分片部分和均逐位一致。用原 kernel 的计算体复测 11 个位置时，优先派发 K 分片的 FFN 输入耗时从 0.7921 增至 0.8026 ms，FFN 输出从 0.3876 降至 0.3863 ms，DeltaNet 输入从 0.4254 增至 0.4360 ms。在线配置保留原排布。两轮峰值均为 3.49 GiB，系统可用内存最低 72%。见 `runs/split-major-probe.log` 和 `runs/split-major-isolate.log`。

用 4K/1K 基准已确认的 token 做 102 段、每段 10 步的离线诊断，两组 MTP 每一步都喂入真实上一 token。复用 MTP 自己的隐状态时，第 1 和第 10 步的首选命中分别为 86/102 和 44/102，平均连续命中 3.41 个候选。每步用主模型真实隐状态及相应 KV 时，第 10 步命中 88/102，平均连续命中 5.34 个。后者依赖主模型事先算出的状态，用来量化递归隐状态的影响。在线验证仍为 278 轮，1024 个输出 token 哈希与无诊断对照一致。日志内总耗时包含 4.37 s 离线评分，整段峰值 55.46 GiB，系统可用内存最低 44%。见 `runs/mtp-depth-probe.log`。

复测保留原词表投影归一化，将归一化前的 MTP 隐状态传给下一个草拟步。平均连续命中降至 2.72 个，第 10 步首选命中降至 15/102。同场两组原始诊断计数完全复现。该变体仅在离线评分中运行，原验证路径仍为 278 轮，整段峰值 55.46 GiB，系统可用内存最低 44%。见 `runs/mtp-prenorm-probe.log`。

在同一段 4K/1K 输出上，用前序已验证的主模型状态和 MTP 状态估计逐通道校正。每段十步开始前冻结校正量，段内仍以真实 token 逐步喂入 MTP。102 段平均连续命中为原始 3.41、加性偏差校正 2.95、正则化逐通道线性校正 2.79。两种校正均只在离线诊断中运行，原验证路径的 token 哈希和 278 轮验证复现。日志计时包含 7.50 s 离线评分。整段峰值 55.46 GiB，系统可用内存最低 44%。见 `runs/mtp-calibration-probe.log`。

双分叉紧凑验证通过 16 节点状态逐位校验和整模分支隔离检查。4K/1K 同场对照有 26 轮启用了两个分叉点，确认的备选候选仍为 41 个，1024 个输出 token 一致，主模型都验证了 287 轮。草拟耗时从 4.83 增至 4.98 s，解码从 30.02 降至 29.85 token/s。两组整段峰值均为 55.46 GiB，系统可用内存最低 44%。双分叉保留为命令行实验选项。见 `runs/multi-fork-ab.log`。

复用原版主模型最后一层，在每段十步草拟开始时截取已验证前缀的 KV。抽取八处，用真实倒数第二层状态重放，归一化末层输出与记录逐位一致。换成 MTP 隐状态的近似输入后，混合末层输出再递归的平均连续命中为 3.18 个，直接用末层输出评分为 2.74 个，原始 MTP 为 3.41 个。末层计算和草拟词表评分平均 2.31 ms/位置。变体仅在离线诊断执行，日志总计时含 6.94 s 离线评分，1024 个输出 token 和 278 轮验证复现。整段峰值 55.46 GiB，系统可用内存最低 44%。见 `runs/mtp-tail-probe.log`。

上下文续写分支用最近 8192 个已知 token 的主机缓冲，匹配 6、8、12 或 16 token 的后缀，再把先前续写作为 MTP 验证树的附加分支。4K/1K 对照在 11 轮加入 55 个节点，命中 2 轮，确认 6 个候选。主模型验证从 278 增至 285 轮，解码由 31.15 降至 30.33 token/s，输出 token 哈希有变化。两组整段峰值均为 55.46 GiB，系统可用内存最低 44%。该分支保留在实验夹具中，线上沿用原配置。见 `runs/context-branch-ab.log`。

完整词表融合使用原有八组 FP32 累加顺序，顶层仍按 BF16 分数选全词表最大值。2、5、8、11、13、16 个验证位置的 BF16 分数逐位一致，索引和并列顺序一致。11 个位置的词表投影从 3.94 降至 3.52 ms。单模型驻留的 4K/1K 对照为 31.18 和 31.43 token/s，两组 1024 个输出 token 逐个一致、均验证 278 轮。解码峰值从 54.53 降至 54.43 GiB，整段峰值 55.46 GiB。见 `runs/packed-head-fused-order.log` 和 `runs/packed-head-fused-ab.log`。

正式实现通过 2–16 个验证位置的完整词表 argmax 检查。启用后，同一 4K/1K HTTP 请求实测解码 32.87 token/s、响应 47.39 s，生成文本哈希与先前 32.47 token/s 的 HTTP 运行一致，均验证 266 轮。解码峰值 54.44 GiB，整段峰值 55.46 GiB，系统可用内存最低 44%。短请求、超预算拒绝和请求后健康检查均通过。见 `runs/http-fused-head.log`。

两层真实权重的微基准中，把 gate/up 分配给不同 SIMD 组会拖慢 FFN。同一 SIMD 组复用输入片段后，11 个验证位置的完整 FFN 从 1.11 降至 0.93 ms，SwiGLU 片段逐位一致。9–16 位置启用该路径的 4K/1K 同场对照为 32.02 对 31.33 token/s，主模型累计验证耗时从 27.76 降至 27.19 s，均验证 278 轮且 1024 个输出 token 一致。解码峰值从 54.43 降至 54.37 GiB，整段峰值 55.46 GiB。见 `runs/ffn-shared-swiglu-probe.log`、`runs/ffn-paired-swiglu-probe.log` 和 `runs/ffn-paired-swiglu-ab.log`。

正式 FFN 实现通过 9–16 个验证位置的 SwiGLU 片段逐位检查。相同 4K/1K HTTP 请求实测 32.99 token/s，生成文本哈希与上一版一致，均验证 266 轮。解码峰值 54.38 GiB，整段峰值 55.46 GiB，系统可用内存最低 44%。健康检查、短请求和超预算拒绝均通过。见 `runs/http-paired-swiglu.log`。

单份词表配置直接从重排后的 BF16 权重读取任意候选行。正式校验期间逐块回读，与原始 checkpoint 逐位比对，首 token 的分数也逐位一致。独立微基准的首 token 投影从 3.62 增至 17.69 ms，分块临时矩阵最大约 159 MiB。4K/1K 同场 CLI 对照中，常驻内存从 53.26 降到 50.89 GiB，整段峰值从 55.46 降到 53.09 GiB，解码峰值从 54.37 降到 52.00 GiB。两组生成的 1024 个 token 逐个一致，均验证 278 轮，解码分别为 31.59 和 32.31 token/s。当前复测命令为 `.venv/bin/python scripts/tune.py dualhead10 compacttree10`。见 `runs/packed-head-single-probe.log` 与 `runs/packed-head-single-ab.log`。

部署单份词表后，同一 4K/1K HTTP 请求实测解码 33.15 token/s，首 token 延迟 16.26 s，生成文本哈希与双份词表 HTTP 对照一致，均验证 266 轮。解码峰值 52.01 GiB，整段峰值 53.09 GiB，系统可用内存最低 45%。短请求、超预算拒绝和健康检查均通过。见 `runs/http-packed-head-single.log`。

两层真实权重的 FFN down 投影复现了 34 组部分和的归约顺序。把第 33、34 组先并入前两 lane，再做 32-lane SIMD 求和，2、5、8、11、13、16 个验证位置的输出均逐位一致。11 个位置时，完整线程组暂存从原路径 0.396 ms 增至 0.574 ms，只暂存实际 token 并提前合并后复测为 0.484 ms，同场原路径为 0.395 ms。两轮峰值 1.33 GiB，系统可用内存最低 72%。线上保留原 down 投影。见 `runs/ffn-down-local-probe.log` 和 `runs/ffn-down-compact-probe.log`。

DeltaNet 线程组内归约和因果卷积在两层真实权重上逐位复现 QKV、z 和回滚用历史，覆盖 2–16 个位置的线性与分叉形状。11 个分叉位置仅融合卷积时从 0.356 增至 0.369 ms。一起融合 a/b 投影后，从含原 a/b 投影的 0.385 降到 0.365 ms。a/b 在实测中有少量 BF16 舍入差异，最大绝对差为 0.000244。4K/1K 整模平均每轮验证从 96.84 降至 95.56 ms，输出 token 哈希发生变化，验证从 278 增至 286 轮，解码从 32.35 降至 32.01 token/s。融合组整段峰值 53.14 GiB，系统可用内存最低 45%。线上保留原路径。见 `runs/packed-delta-local-conv-probe.log`、`runs/packed-delta-all-inputs-retest.log` 和 `runs/packed-delta-all-inputs-ab.log`。

复现 MLX 0.32.4 对 a/b 投影的 32 个等长 K 分片，再按分片顺序做 FP32 累加。两层正式权重、2–16 个位置的 233,280 个 BF16 结果逐位一致。完整融合在 11 个位置的微基准从 0.388 降至 0.377 ms。4K/1K 同场对照中，1024 个输出 token 和 278 轮验证一致，累计验证时间从 26.90 增至 27.25 s，解码从 32.37 降至 31.98 token/s。融合组峰值 53.14 GiB，系统可用内存最低 45%。线上保留原路径。见 `runs/delta-ab-exact-probe.log` 和 `runs/delta-ab-exact-fused-ab.log`。[MLX 0.32.4 Metal matmul 源码](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/matmul.cpp)

宽批次 FFN 提速后复测 10、12、15 候选上限。三组 4K/1K 输出 token 一致，分别验证 278、275、275 轮，草拟打分分别为 2055、2121、2255 次。首次顺序测试的首 token 延迟从 16.17 漂移至 22.71 s，吞吐差异受运行时序影响。同一模型驻留的反向复测中，12 与 10 候选的首 token 延迟接近，为 29.37 和 29.82 s，解码分别为 16.04 和 16.86 token/s。保留 10 候选线上配置。各组峰值约 53.09 GiB，系统可用内存最低 45%。见 `runs/wide-depth-retest.log` 和 `runs/wide-depth-reverse.log`。

随后使用当前驻留服务复测 4096/256 HTTP 请求，首 token 为 16.28 s，解码为 32.98 token/s。空闲和请求期间的采样中，累计换入、换出和 pageout 计数均保持不变。该次复测未定位先前计时漂移的原因。见 `runs/host-pressure-idle.log` 和 `runs/host-pressure-http.log`。

加载器用一个 Metal kernel 将 BF16 权重直接写入 SIMD 排布。真实词表矩阵的排布与旧实现逐位一致，回读保持原始权重位模式。词表重排单项峰值由 9.47 降到 4.74 GiB，耗时中位数从 211 降到 80 ms。整模加载对照峰值从 58.00 降到 53.26 GiB，耗时从 4.42 降到 3.12 s。运行时模型常驻仍为 50.89 GiB。见 `runs/direct-pack-probe.log` 和 `runs/http-direct-pack.log`。

更新加载器后，同一 4K/1K HTTP 请求实测 33.49 token/s，生成文本哈希与上一版一致，均验证 266 轮。请求峰值 53.09 GiB，系统可用内存最低 45%。本次改动作用于加载阶段。kernel 校验、整模校验和请求后健康检查均通过。

扩大首位置备选分差的实验保留 1.0 内最多两个备选节点，并在分差 1.0–3.0 之间只加一个备选节点。两次同模型驻留对照均在新增区间提交 65 轮并命中 10 轮，整段验证从 278 降至 274 轮，输出 token 哈希变化且可复现。两组交换顺序后，吞吐分别为 32.53 对 32.34、31.98 对 30.04 token/s。后一组对照的首 token 延迟也增加了约 0.88 s，计时收益幅度尚不稳定。各组峰值约 53.09 GiB，系统可用内存最低 45%。线上仍使用 1.0 分差。见 `runs/expanded-root-branch-ab.log` 和 `runs/expanded-root-branch-reverse.log`。

在 4K KV 前缀的微基准中，两种合并排布和单独预留均逐位复现注意力结果与分支提交。单 token 的追加、单独预留和合并预留耗时分别为 0.146、0.112 和 0.100 ms。4K/1K 整模对照只对 MTP 单 token 启用单独预留。草拟耗时从 4.54 降到 4.34 s，解码为 33.09 对 32.33 token/s，1024 个输出 token 和 278 轮验证一致，峰值均为 53.09 GiB。见 `runs/kv-layout-probe.log` 和 `runs/kv-layout-reserve-ab.log`。

正式 MTP 缓存实现通过 512→1024 容量增长、原分叉快照和单 token 输出逐位校验。部署后，同一 4K/1K HTTP 请求实测 32.80 token/s，草拟耗时约 4.40 s，与上一份 HTTP 记录接近。两次生成文本哈希一致，均验证 266 轮。当前运行峰值 53.09 GiB，系统可用内存最低 45%。健康检查和超预算拒绝通过。见 `runs/http-mtp-kv-reserve.log`。

使用 HTTP 的同一 4096-token 对话输入和 EOS 条件，在单份模型驻留内对照 MTP 合并与单独预留 KV。两组输出 1024 个 token 且逐个一致，均验证 266 轮。合并使草拟耗时从 4.185 降至 4.148 s，解码从 34.52 升至 34.57 token/s。两组峰值 53.09 GiB，系统可用内存最低 45%。合并实现通过跨容量边界和分支快照的逐位检查，收益量级有限，运行时沿用单独预留。以单独预留对照计算，50 token/s 需要平均每轮确认约 5.57 个 token，或者把当前 266 轮的平均耗时从 111.4 降到 76.9 ms。见 `runs/mtp-kv-combined-http-ab.log`。

四片段无损排布将 64 个 FFN down 矩阵缩到原始字节数的 76.18%，每个矩阵均通过逐位回读。整模 4K/1K 对照的常驻从 50.89 降至 48.36 GiB，整段峰值从 53.09 降至 50.56 GiB，解码从 33.17 至 33.24 token/s。两组输出的 1024 个 token 逐个一致，均验证 278 轮。两层微基准中，6 个位置的 down 投影从 0.371 降至 0.337 ms，11 个位置从 0.397 增至 0.421 ms。复测命令为 `.venv/bin/python scripts/tune.py plainffndown10 compacttree10`。见 `runs/bf16-packet-retest.log` 和 `runs/bf16-down-lossless-ab.log`。

正式无损 down 配置验证了 64 层权重，4K/1K HTTP 实测 34.83 token/s，生成文本哈希与上一版一致，均验证 266 轮。请求峰值 50.56 GiB，解码峰值 49.48 GiB，系统可用内存最低 47%。kernel、整模、超预算拒绝和健康检查均通过。见 `runs/http-lossless-down.log`。

gate/up 联合编码复用一个包描述符和连续载入。两层真实权重的 SwiGLU 片段逐位一致，6 个位置耗时从 0.752 降至 0.545 ms，11 个位置从 0.679 降至 0.665 ms。4K/1K 同场对照逐层回读 64 个矩阵，常驻从 48.36 降至 43.38 GiB，整段峰值从 50.56 降至 45.58 GiB。解码为 34.81 对 33.74 token/s，均验证 278 轮，1024 个输出 token 逐个一致。复测命令为 `.venv/bin/python scripts/tune.py plaingateup10 compacttree10`。见 `runs/bf16-gate-pair-probe.log` 和 `runs/bf16-gate-joint-ab.log`。

联合编码正式部署后，同一 4K/1K HTTP 请求实测 36.40 token/s，先前仅编码 down 的记录为 34.83 token/s。两次生成文本哈希一致，均验证 266 轮。当前解码峰值 44.50 GiB，整段峰值 45.58 GiB，系统可用内存最低 50%。加载及逐位校验约 13.37 s。内存下降后，245,760 token 上限的短提示词请求可通过工作集预算并在 EOS 停止。262,000 上限的探针仍返回 HTTP 503。见 `runs/http-joint-gate-up-retest.log` 和 `runs/http-joint-gate-up-final.log`。

两类注意力输入联合编码在真实权重的 2、6、11、13、16 个验证位置上逐位复现投影。11 个位置的 DeltaNet 输入从 0.404 降至 0.343 ms，全注意力从 0.368 降至 0.313 ms。4K/1K 整模对照逐位回读 64 个矩阵，累计验证耗时从 24.92 降至 24.11 s，解码为 35.80 对 34.79 token/s。1024 个输出 token 一致，均验证 278 轮。常驻从 43.38 降至 41.11 GiB，整段峰值从 45.58 降至 43.31 GiB。复测命令为 `.venv/bin/python scripts/tune.py plaininputs10 compacttree10`。见 `runs/bf16-paired-input-probe.log` 和 `runs/bf16-lossless-inputs-ab.log`。

线上请求验证了 64 个输入矩阵、4096/1024 token、以及与上一版一致的文本哈希和 266 轮验证。峰值 43.31 GiB，系统可用内存最低 50%，健康检查和模型上下文上限验证通过。这次 HTTP 的首 token 延迟为 19.48 s，解码为 20.37 token/s，未复现同场 CLI 的提速。随后同一驻留服务的 4096/256 短诊断为 23.21 token/s，采样期间累计换页和交换计数保持不变。当前计时漂移原因尚未定位，线上保留经过整模对照验证的输入编码。见 `runs/http-lossless-inputs.log` 和 `runs/host-pressure-lossless-inputs.log`。

注意力输出矩阵联合编码逐位匹配原投影，64 层 4K/1K 对照常驻从 41.11 降至 40.23 GiB，峰值从 43.31 降至 42.43 GiB。两组输出 token 逐个一致并均验证 278 轮，解码从 36.43 降至 36.17 token/s。该配置保留为可选项。完整 248,320 行词表的候选路径将权重缩至 76.53%，全矩阵回读和 32,774 行任意候选读取均逐位一致，BF16 分数、argmax 和并列索引通过检查。6 个位置的词表投影从 3.49 降至 2.85 ms，11 个位置从 3.49 降至 3.43 ms。完整词表方案仍在微基准阶段。见 `runs/bf16-lossless-outputs-ab.log`、`runs/bf16-remaining-proj-probe.log` 和 `runs/bf16-full-head-probe.log`。

使用现有输入编码配置复测同一 4K/1K HTTP 请求，解码为 37.97 token/s，首 token 为 16.24 s。文本哈希与此前一致，均验证 266 轮。峰值 43.31 GiB，系统可用内存最低 51%，服务健康检查通过。此前记录的 20.37 token/s 降速在这次复测中未出现。见 `runs/http-lossless-inputs-recheck.log`。

完整词表编码集成后，整模校验逐位核对全部词表权重、随机行和首 token logits，2–16 个位置的 argmax 也一致。4K/1K 同场对照的解码为 35.51 → 35.96 token/s，常驻内存为 41.11 → 40.55 GiB，请求峰值为 43.31 → 42.75 GiB。两组逐个匹配全部 1024 个输出 token，均验证 278 轮，系统可用内存最低均为 50%。见 `runs/bf16-lossless-vocab-ab.log` 和 `runs/http-lossless-vocab.log`。

线上启用词表编码后，同一 4096/1024 HTTP 请求的解码为 38.10 token/s，首 token 为 16.12 s，文本哈希与上一版相同，均验证 266 轮。请求峰值 42.75 GiB，解码峰值 41.63 GiB，系统可用内存最低 50%。短请求、模型上下文上限和请求后健康检查均通过。见 `runs/http-lossless-vocab.log`。

对当前无损权重布局测试 Metal 原生 BF16 操作数。FFN、DeltaNet 输入和完整词表的 2、6、11、16 个位置通过逐位比对。11 个位置的 FFN down 用四 SIMD 组从 0.358 降至 0.343 ms，六个位置最快为 0.261 ms，原路径为 0.262 ms。其余投影的改变量较小。该轮峰值 9.73 GiB，系统可用内存最低 69%。见 `runs/bf16-native-packets-probe.log`。

在三层 FFN down 的完整原始矩阵上，三位指数主体配合稀疏原始指数把存储量从 76.15%–76.20% 降至 70.65%–71.07%。所有 BF16 权重和 FP32 部分和逐位一致。6 个位置从 0.272 增至 0.740 ms，11 个位置从 0.361 增至 1.038 ms。该编码会在热循环中为稀疏指数计算 SIMD 前缀位置并查表，线上继续使用当前四位指数布局。试验峰值 4.11 GiB，系统可用内存最低 70%，服务恢复后通过健康检查。见 `runs/bf16-exponent-bands.log` 和 `runs/bf16-three-bit-probe.log`。

四层真实 FFN down 权重的 2–16 位置微基准中，两个输出块共用激活片段，FP32 部分和逐位一致。四 SIMD 组的实现将 6 个位置从 0.262 降至 0.251 ms，将 11 个位置从 0.356 降至 0.288 ms。4K/1K 同场对照的解码从 34.99 升至 36.01 token/s，累计主模型验证耗时从 24.76 降至 23.96 s。两组均验证 278 轮，1024 个输出 token 逐个一致。常驻内存从 40.55 增至 40.59 GiB，请求峰值从 42.75 增至 42.79 GiB，系统可用内存最低均为 50%。见 `runs/bf16-paired-down-probe.log` 和 `runs/bf16-paired-down-ab.jsonl`。

成对 FFN down 的整模和 HTTP 输出校验通过。首次 HTTP 复测为 37.66 token/s。随后用同一份 4096/1024 HTTP 输入连续切换两个布局，原布局测得 38.01 token/s，成对布局测得 25.63 token/s，首 token 延迟同时从 16.27 增至 19.75 s。两种布局输出哈希相同，均验证 266 轮，系统可用内存最低均为 51%。计时波动的原因尚未定位。服务恢复原布局，成对布局保留为 CLI 选项。见 `runs/http-paired-down-ab.log` 和 `runs/http-paired-down-restored.log`。

按真实 4096/1024 HTTP 请求的已确认前缀回放 MTP。在 1023 个起点中，真实下一 token 有 863 次位列草拟第一、77 次第二、29 次在草拟词表外。现行配置实际验证 266 轮。允许提前知道匹配结果时，回放估计的验证轮数下界为线性 10 候选 268、线性 15 候选 263、仅根部允许第二候选 237、任意深度允许第二候选 194。该估计以单位置 MTP 回放为依据，额外候选计算和在线分支选择仍需实测。回放过程中峰值 41.67 GiB，系统可用内存最低 51%。见 `runs/mtp-rank-oracle.log` 和 `runs/mtp-rank-oracle-topology.json`。

16 节点紧凑验证树允许第二个分叉点位于备选链内，DeltaNet 仍使用原有两个寄存器状态槽。Metal 测试逐位匹配树上每个节点的状态与输出。整模测试中更改非祖先节点 token 时分支隐藏状态逐位一致，提交嵌套分支后的下一 token 与线性路径一致，logits 平均绝对差为 0.01443。在线草拟仍沿用现有分支策略。见 `runs/nested-tree-verify.log`。

用同一份 4096/1024 HTTP 提示词与同一份驻留模型测试在线分支规划。现行策略为 266 轮和 38.13 token/s；嵌套分支为 273 轮和 29.68 token/s；主链双分支为 241 轮和 33.42 token/s。后两者的生成 token 与对照有差异。嵌套分支构造 173 次，实际选中 5 次。超过 8 个位置的验证轮数分别为 113、269、239，后两种方案的主模型单轮均值从 84.74 增至 102.69 和 103.33 ms。峰值均为 42.75 GiB，系统可用内存最低 50%。服务沿用现行策略。见 `runs/mtp-nested-draft-ab.log`。

在 4096 KV 前缀上，对两条和三条草拟链各跑六步，共享原始 MTP 的投影、注意力和词表打分。两链从 22.18 降至 21.55 ms，三链从 31.42 降至 22.86 ms。30 个位置的草拟 argmax 与串行一致，整行 logits 有 BF16 数值差异。该试验只加载 MTP 与所用词表，峰值 6.62 GiB，系统可用内存最低 74%。见 `runs/mtp-branch-batch-probe.log`。

固定节点预算试验只在窄轮次重分配槽位，以累计 MTP 概率决定保留主链尾部或添加备选链。对照与试验的 4K/1K 解码为 38.11 与 36.40 token/s，验证 266 与 272 轮。窄轮加入备选 157 次、命中 20 次，主链尾部共替换 8 个节点。试验第 7 个输出 token 首次与对照不同。对照中 6、8、11 个验证位置的主模型单轮均值分别为 76.54、83.77、93.93 ms。请求峰值均为 42.75 GiB，系统可用内存最低 50%。见 `runs/mtp-compact-budget-ab.log`。

回放该请求的前 16 个输出，定位第 7 个 token 的分歧。原策略在 8 位置验证批次中得到两个 21.75 的 BF16 logits，选较小的 token ID 262。实验策略在 11 位置批次中得到 21.75 与 21.875，选择 ID 285。沿共同前缀回看，最早观察到的隐藏状态差异出现在第 4 层全注意力层，最大差为 0.00098。尚未隔离到具体算子。回放峰值 42.76 GiB，服务恢复后通过健康检查。见 `runs/mtp-verify-drift.log`。

进一步采样第 4 层两个批次中的同一逻辑位置，Q、K、V 逐位一致，原注意力输出最大差为 0.003906。实验 Metal 内核按逻辑位置分块读取共享前缀及分支祖先，两个批次的该位置输出逐位一致。两份真实 8 位置样本的单层耗时从 0.552/0.528 降到 0.501/0.500 ms。8、11、16 位置树中抽样的节点也与单位置计算逐位一致。见 `runs/attention-logical-path-ab.log`。

相同 4K/1K 请求中，原注意力为 38.16 token/s、266 轮。逻辑位次注意力下，现行草拟策略与固定节点预算分别为 37.20 与 37.43 token/s、271 与 265 轮，两者生成的 1024 个 token 完全相同，与原注意力的首次差异出现在第 319 个输出 token。三组峰值均为 42.75 GiB，系统可用内存最低 51%。线上保留原注意力，健康检查通过。见 `runs/logical-attention-request-ab.log`。

实权重对照了原稠密包、固定地址、固定地址合并载入，以及按 K 分片顺序读取。各布局恢复的 BF16 权重和短块结果均逐位一致。固定地址使 gate/up 存储从约 76.5% 增到 78.5%–79.0%，6 位置从 0.478 增至 0.501 ms。只额外保留分片起始地址的顺序读取为 0.479 ms。其余小幅收益不足以启用整模改动。试验峰值 7.19 GiB，系统可用内存最低 70%。见 `runs/bf16-fixed-slots-probe.log`。

原始 MTP 层与 32K 草拟词表的无损编码逐位回读全部权重，编码大小为原始的 76.52%。在 4096 KV 前缀上的八步草拟，原布局、仅编码词表、同时编码 MTP 投影分别为 16.22、15.37、13.39 ms，八步 argmax 均一致。仅编码词表的最大单步 logits 平均绝对差为 0.0000011，同时编码 MTP 为 0.03485。试验峰值 6.62 GiB，系统可用内存最低 74%。见 `runs/mtp-lossless-short-probe.log`。

在同一份驻留模型上复测 4096/1024 请求。原布局、仅编码草拟词表、同时编码 MTP 单步投影分别为 38.08、38.33、39.04 token/s，草拟用时分别为 4.183、4.033、3.620 s。仅编码草拟词表时，266 轮验证和 1024 个输出 token 与原布局一致，解码峰值从 42.48 降到 42.17 GiB。同时编码 MTP 时验证为 263 轮，第 960 个输出 token 首次变化。对照额外驻留 0.84 GiB 的编码数据，三组整段峰值均为 43.60 GiB，系统可用内存最低 49%。服务沿用当前配置并通过健康检查。见 `runs/mtp-lossless-request-ab.log`。

正式部署 32,768 行草拟词表无损编码后，同一 4K/1K HTTP 请求从 38.09 升至 38.38 token/s，草拟用时从 4.178 降至 4.028 s。两组文本哈希一致，均验证 266 轮。整段峰值从 42.75 增至 42.99 GiB，解码峰值从 41.63 降至 41.56 GiB，系统可用内存最低 50%。整模、短请求和健康检查通过。见 `runs/http-lossless-draft-vocab-ab.log`。

单份共享词表微基准将固定草拟范围的两半排入同一段编码包，草拟只切取描述符视图。全部 248,320 行按原 token ID 顺序逐位回读，单步草拟分数、2/6/11/16 位置的主模型分数和 argmax 与分开编码逐位一致，并列时仍选最小 token ID。共享存储从 2,202,743,552 降至 1,946,050,048 字节，另有 2.5 MiB 描述符切片。草拟投影从 0.587 降至 0.579 ms，主词表投影差异小于 0.005 ms。试验峰值 11.78 GiB，系统可用内存最低 68%。见 `runs/shared-draft-vocab-probe.log`。

共享词表正式部署后，同一 4K/1K HTTP 对照从 37.91 升至 38.41 token/s，输出文本哈希相同，均验证 266 轮。请求峰值从 42.99 降至 42.75 GiB，解码峰值从 41.56 降至 41.32 GiB，系统可用内存最低均为 50%。整模逐位回读、首 token、草拟打分、跨分区并列和服务健康检查均通过。见 `runs/http-shared-draft-vocab-ab.log`。

四层真实 FFN down 上试验四个输出块共用激活。1–16 位置测试覆盖的 FP32 部分和与原路径逐位一致。11 个位置的原路径与两块共享分别为 0.360 和 0.289 ms，四块最快为 0.351 ms。6 个位置的四块最快路径为 0.318 ms，两块为 0.252 ms。继续使用可选的两块共享内核。微基准峰值 2.45 GiB，系统可用内存最低 71%。见 `runs/bf16-four-tile-down-probe.log`。

在同一份驻留模型上逐层切换 FFN down 布局并切回。三次 4K/1K 解码为 38.36、38.59、38.29 token/s，验证用时为 22.56、22.38、22.58 s，输出逐 token 一致，均验证 266 轮。两个切换过程只暂存当前层的解包和新布局，峰值分别为 42.34 和 42.35 GiB。请求峰值分别为 42.75、42.79、42.75 GiB，系统可用内存最低均为 50%。直接在原编码上共用激活的独立双块内核也逐位匹配，11 个位置为 0.322 ms，同场原路径和联合编码双块为 0.375 与 0.305 ms。见 `runs/paired-down-resident-aba.log` 与 `runs/bf16-independent-pair-down.log`。

两块共享与共享词表一起通过内核、整模、短请求和上下文上限校验。4K/1K HTTP 样本为 37.68 token/s，仍验证 266 轮且文本哈希一致。请求峰值 42.79 GiB、系统可用内存最低 50%。该次 HTTP 未复现驻留对照中的提速，服务已恢复主线布局。可用 `.venv/bin/python scripts/tune.py shareddraft10 sharedpaireddown10` 复测。见 `runs/http-shared-paired-down.log`。

恢复后的独立 HTTP 样本为 34.68 token/s，首 token 16.57 s，请求峰值 42.75 GiB，系统可用内存最低 52%。输出哈希和 266 轮验证一致，短请求与健康检查通过。服务重启前后的计时波动仍大于驻留对照观察到的约 0.3 token/s 布局差异。见 `runs/http-shared-paired-down-restored.log`。

四层真实权重的短块 FFN 静态形状编译逐位匹配 BF16 输出。6、11、16 位置的建图耗时由约 10.2–10.7 降至 5.2–5.4 微秒，含 GPU 执行的单层耗时由 0.634、0.800、0.808 降至 0.627、0.793、0.799 ms。仅编译 FFN 尾部的收益更小。无形状编译遇到 Metal CustomKernel 无法推导输出形状的限制。微基准峰值 5.16 GiB。见 `runs/compiled-short-ffn-probe.log`。

同一驻留模型的对照、仅编译 FFN、FFN 加 6/11 位置 DeltaNet 冷缓存和热缓存分别为 38.28、38.32、38.69、38.66 token/s。四组逐个匹配全部 1024 个输出 token，均验证 266 轮，峰值均为 42.75 GiB。冷缓存给两个宽度各建立 47 个新图，热缓存无新增。编译的线性 DeltaNet 逐位匹配 BF16 输出、FP32 状态和指定前缀的回滚。见 `runs/compiled-verify-request-ab.log`。

正式配置为 6 和 11 位置的线性 DeltaNet 缓存图。相同 4K/1K HTTP 对照从 38.28 升至 38.52 token/s，累计验证耗时从 22.59 降至 22.41 s。文本哈希一致，均验证 266 轮。整段峰值均为 42.75 GiB，解码峰值均为 41.32 GiB，系统可用内存最低为 50%。整模和回滚逐位校验、短请求和健康检查均通过。见 `runs/http-compiled-delta-ab.log`。

用同一 4K/1K 请求测试根候选接近时最多 8 个验证位置的多链草拟。批量展开将 451 次独立 MTP 前向合并为 309 次，草拟耗时从 4.776 降到 4.155 s。串行与批量建树都验证 287 轮，1024 个输出 token 相同，解码从 35.32 提到 36.18 token/s。两种建树均从第 319 个输出 token 开始偏离原策略，原策略为 266 轮及 38.69 token/s。三组峰值均为 42.75 GiB，服务保留现行策略。见 `runs/mtp-parallel-beam-ab.log`。

保持原建树规则，只合批根部主链与备选链的第一步。4K/1K 对照复用 56 次备选打分，1024 个输出 token 逐个相同，均验证 266 轮。额外的根部打分读回使合批需要独立同步，草拟耗时从 4.029 增到 4.292 s，解码从 38.60 降至 37.82 token/s。两组峰值均为 42.75 GiB，系统可用内存最低 50%，服务恢复后健康检查通过。见 `runs/mtp-paired-root-ab.log`。

按层 prefill 在原 512-token 分块上复用当前层的行优先权重。520-token 混合长度的每块输出及全层 KV、卷积和 FP32 递推状态均逐位一致。4K/1K 同场对照的首 token 从 16.16 降至 13.60 s，端到端从 24.01 升至 25.49 token/s，峰值从 42.75 降至 41.93 GiB。两组输出逐 token 一致，均验证 266 轮。1024 分块虽将首 token 降到 12.89 s，但第 77 个输出 token 首次变化，验证增至 299 轮。见 `runs/layer-first-prefill-ab.log`。

正式路径通过 520-token 分块逐位校验。独立 HTTP 对照中，按层 prefill 的首 token 从 16.29 降至 15.72 s，同次解码由 38.06 降至 33.86 token/s。随后在同一 HTTP 进程做原路径、按层、原路径对照，首 token 分别为 17.46、17.53、22.12 s，解码为 33.63、28.79、20.88 token/s。各次输出哈希和 266 轮验证一致。按层路径稳定把峰值从 42.75 降至 41.93 GiB，系统可用内存最低 50%–51%。测试前后 pageout 和 swap 计数相同，计时波动原因待定位。服务保持默认 prefill，按层路径保留为 CLI 选项。见 `runs/http-layer-prefill-ab.log` 和 `runs/http-layer-prefill-resident-aba.log`。

草拟词表覆盖试验中，现行 32K 加提示词短名单之外共有 30 个输出位置，涉及 20 个不同 token，10 个位置属于重复。随请求补入已经确认的词表外 token，验证从 266 降至 261 轮，全部输出逐 token 一致。固定扩大到 64K 将词表外输出位置减至 6 个，但验证增至 284 轮，输出发生变化，解码峰值从 41.32 增至 41.63 GiB。见 `runs/draft-vocab-coverage-ab.log`。

复用同一 4K 前缀状态重新生成 1K 输出。两次原配置分别为 38.68 和 28.89 token/s，动态短名单在两者之间测得 34.71 token/s，仍为 261 轮、输出逐 token 一致。另测最少 12 个匹配 token 的既有后缀，5 次提交共接受 40/50 个草拟 token，验证 265 轮，输出发生变化。四组解码峰值均为 41.73 GiB，pageout 和 swap 计数前后相同。计时波动下没有确认吞吐收益。服务恢复并通过健康检查。见 `runs/draft-reuse-snapshot-ab.log` 和 `runs/suffix-retrieval-audit.log`。

固定负载复测复用一份驻留模型与同一份 4096-token 前缀，连续三轮解码分别为 38.60、38.90、38.92 token/s，空闲 42 s 后为 38.45 token/s。四轮各生成 1024 个相同的 token，均验证 266 轮。每轮前后另测四层真实权重、11 个输入位置的固定 FFN，耗时中位数为 3.406–3.492 ms。macOS 热状态一直为 nominal，pageout、swap-in 和 swap-out 计数没有变化。此次未复现此前的渐进降速，其原因仍未确定。固定负载解码峰值 41.73 GiB，系统可用内存最低 51%，测试结束后服务通过健康检查。见 `runs/fixed-decode-drift.log`。

紧接着复测动态草拟词表，macOS 热状态在所有采样点均为 fair。原配置、动态草拟词表、原配置依次为 33.36、31.02、27.53 token/s。固定 FFN 从首轮前的 3.577 ms 升至末轮后的 5.287 ms。动态词表补入 20 个已确认 token，验证从 266 轮减至 261 轮，三轮的全部输出逐 token 一致。由于两轮基线差异更大，实际吞吐收益仍待确认。三轮峰值均为 41.73 GiB，系统可用内存最低 51%，pageout、swap-in 和 swap-out 计数没有变化。CLI 基准 JSON 现记录每个计时请求前后的 `thermal_before` 和 `thermal_after`，用于比较同一条件下的结果。见 `runs/stable-confirmed-draft-ab.log`。

用 `.venv/bin/python scripts/tune.py compileddelta10` 验证两个热状态字段，4K/1K CLI 样本记录 fair → fair、25.98 token/s、峰值 42.75 GiB、系统可用内存最低 50%。此次 CLI 使用原始 benchmark prompt，验证 278 轮。HTTP 服务恢复后返回 BF16 optimized 且 ready。见 `runs/thermal-bench-smoke.log`。

用线上配置和同一份 4K 前缀状态复放 6、11 个已确认位置，验证预测与已记录的后续 token 逐个相同。完整单轮验证的中位耗时分别为 75.61 和 92.39 ms，单独测量完整词表投影为 3.00 和 3.54 ms。每八层分组独立测量、合计 64 层 FFN 为 41.45 和 52.07 ms，其中单测 gate/up 加 SwiGLU 为 29.13 和 35.71 ms，down 为 14.19 和 19.05 ms。FFN 分项使用固定激活，时间不能直接相加还原端到端耗时。两种宽度的采样热状态均为 nominal，实验峰值 41.47 GiB，系统可用内存最低 51%。见 `runs/current-verification-parts.log`。

gate/up 改用每个线程组 4、5 或 8 个 SIMD 组，每组处理多个原有 K 分片，保持累加和归约顺序。四层原版权重、6、11、16 位置的 SwiGLU 输出逐位一致。6 位置从 0.482 ms 增至最快候选的 0.519 ms，11 位置从 0.610 ms 增至 0.619 ms。保留原线程组排布。实验峰值 4.13 GiB，系统可用内存最低 71%。见 `runs/gate-up-split-residency.log`。

同四层权重上，使用一次整数加法同时还原两个 BF16 指数，6 个位置为 0.477 → 0.479 ms，11 个位置为 0.592 → 0.589 ms，16 个位置为 0.597 → 0.588 ms。另测先加载两个相邻权重包再顺序执行乘加，11 个位置为 0.602 → 0.827 ms。两项的 SwiGLU 输出均逐位一致，系统热状态 nominal，峰值 4.13 GiB。线上保留原内核。见 `runs/gate-up-packed-exp.log` 和 `runs/gate-up-prefetch-two.log`。

按 HTTP 记录的 266 轮和 38.52 token/s，平均每轮含草拟与确认约 99.84 ms。50 token/s 在相同轮时下需要将验证降到约 205 轮以内，或者在相同轮数下将每轮降到 76.92 ms。当前 11 位置验证本身约 92 ms，进一步提高每轮接受数和减少每轮权重读取都值得测量。

重复上下文分叉用提示词和已确认输出统计三 token 后缀。在前 8 个主链位置检查第二候选，仅当它在同一后缀后出现至少两次且超过第一候选的两倍时考虑移动分叉。4K/1K 同前缀驻留对照中，原配置、上下文分叉、原配置依次为 38.66、39.64、38.97 token/s，分别验证 266、259、266 轮。候选触发 7 次后置分叉并命中 2 次，第 698 个输出 token 首次发生变化。热状态均为 nominal，固定 FFN 为 3.27–3.42 ms，峰值 41.75 GiB，系统可用内存最低 51%。见 `runs/context-prior-branch-ab.log`。

复核第 698 个 token 时，两条路径在该轮均使用 11 位置线性验证，发生分歧的输出行分别为 2 和 4（从 0 起计）。上下文分叉的整词表打分选中 token 279，两者得分分别为 264 的 24.750 和 279 的 24.875。使用共同的前 697 个输出 token 重放，尾部 16 或 48 个位置逐 token 前向，两次整词表 argmax 均为 264，得分 24.875，279 的得分为 24.750。该结果定位了候选调度触发的目标打分变化。复核峰值 53.93 GiB，系统可用内存最低 51%。线上保留原草拟策略并通过健康检查。见 `runs/context-prior-branch-precision.log`。

同一分歧点复放时，把上下文分叉的轮起点推进两个 token，分别用原 11 位置验证后回滚、直接用 2 位置前向。两种方法都在与基线相同的轮起点重放相同的 11 个候选，分歧位置的 64 层隐藏状态与原上下文分叉逐位相同，仍选 279。与基线比较时，第 5 层首次出现隐藏状态差异，最大绝对差为 0.000977，入轮前该层的 FP32 DeltaNet 状态已不同。第 64 层平均绝对差为 0.07870。该实验排除了这两个相邻批次边界本身造成此处翻转。峰值 43.42 GiB，系统可用内存最低 50%。见 `runs/context-branch-layer-drift.log` 与 `runs/context-branch-layer-drift.json`。

从共同的入轮缓存分别复放最早三处同起点且候选结构不同的验证轮，绝对位置为 4282、4521、4557。三组分别覆盖 11/13、11/13 和 8/6 个验证位置，沿两条已接受路径共有的 9、7 和 1 个节点，64 层隐藏状态均逐位一致。试验覆盖单轮前向输出，尚未定位跨轮提交状态从何时开始分歧。峰值 43.60 GiB，系统可用内存最低 51%，服务恢复后通过健康检查。见 `runs/draft-first-topology-drift.log`。

从同一份缓存按两条实际验证调度提交已接受 token。首个共同边界状态分歧落在绝对位置 4521–4546。逐层复放将最早的前向差异定位到绝对位置 4528 的第 4 层全注意力。基线在下一轮线性验证中处理该位置，候选在上一轮的后置分支中处理。可见的 Q、K、V 逐位相同，现行注意力输出平均绝对差为 0.00001249、最大 0.00390625。两条路径分别取该查询的真实祖先、单独计算该行时输出逐位一致。两条路径提交同一前缀后，第 5 层 FP32 DeltaNet 状态最先出现差异。峰值 42.66 GiB，系统可用内存最低 51%。见 `runs/draft-commit-drift.log`。

只对后置分叉节点按实际祖先拼接 KV 并逐行计算注意力。4K/1K 驻留对照的 1024 个输出 token 全部匹配原策略，包括第 698 个 token。原配置、后置分叉实验、原配置依次为 38.58、38.70、38.81 token/s，均为 266 轮。实验原先的 259 轮结果来自不同输出序列，同输出条件下未测得缩减。峰值 41.76 GiB，系统可用内存最低 51%。末轮基线结束时热状态为 fair，其余采样为 nominal。线上保留原策略并通过健康检查。见 `runs/context-compact-tail-ab.log`。

动态草拟词表再次做顺序对照时，两轮基线从 38.18 降至 34.11 token/s，候选居中为 37.64 token/s。改用两份状态缓存共用一份 BF16 权重，按验证轮交替推进到同一输出进度。计入各自的草拟、验证和提交时间后，基线为 26.52 token/s，候选为 27.08 token/s，提升 2.13%。候选补入 20 个已确认 token，验证由 266 降为 261 轮，1024 个输出逐个一致。热状态均为 fair，两路共同峰值 42.26 GiB，系统可用内存最低 51%，换页和交换计数不变。见 `runs/confirmed-draft-nominal-ab.log`、`runs/confirmed-vocab-interleaved.log`。

`--draft-confirmed-vocab` 将已经由主模型确认的输出 token 补入请求内的草拟词表，最多追加 128 行。权重行从原版 BF16 词表提取，并按 token ID 排序。主模型继续用完整 BF16 词表验证。重排前按现有 Metal 内存预算检查临时分配。优化 HTTP 配置已启用此项。可用 `.venv/bin/python scripts/tune.py compileddelta10 confirmedvocab10` 对照。

标准 CLI 入口以不同的原始代码输入完成 4096/1024 测试，补入 12 行。输入和输出 token 哈希与同输入的既有基线一致，均为 278 轮。此次候选采样热状态为 fair → fair，解码 21.43 token/s，峰值 42.75 GiB，系统可用内存最低 50%。该结果验证了入口和输出，吞吐增益仅由同条件交替对照支持。优化 HTTP 服务恢复后通过 BF16 健康检查，默认草拟词表策略保持原配置。见 `runs/confirmed-vocab-native.log`。

给已记录的 MTP 真前缀排位回放加入单轮成本，窄/宽验证分别使用实测 75.61/92.39 ms，草拟每次打分使用 HTTP 样本平均的 2.11 ms。假设预知匹配长度、无需为被丢弃的侧链额外付费，线性至 15 个候选的最短估算用时为 21.83 s（263 轮、46.86 token/s）。允许在任意深度使用一次分差不超过 1.0 的第二候选时为 20.16 s（240 轮、50.74 token/s）。这些是该输入的离线估算，在线分叉选择、实际批宽细节和回滚仍有成本。见 `runs/mtp-time-weighted-oracle.log`。

同一份驻留模型的 4K/1K HTTP 对照中，基线、动态词表、基线分别为 35.61、35.27、35.09 token/s，验证 266、261、266 轮。三次文本哈希相同，候选补入 20 个 ID。前两次固定 FFN 前后分别为 3.502 → 4.407 ms、3.501 → 4.325 ms。最后一次请求的生成指标完整返回，对照进程退出前未写完最后一次主机采样。三次峰值 42.75 GiB，系统可用内存最低 51%。HTTP 吞吐未复现交替对照的提速，继续保留 CLI 候选和原线上配置。优化服务恢复后通过健康检查。见 `runs/confirmed-vocab-http-ab.log`。

验证宽度限制为 8 个位置，保留 5 步检查点，若根部分叉成立则在 5 个主链候选外预留 2 个分叉位置。4K/1K 按验证轮交替测试时，基线有 113 轮超过 8 位置，候选全部在 8 以内。平均主模型单轮耗时从 84.32 降至 77.25 ms，草拟总耗时从 4.03 降至 3.67 s。候选验证轮数从 266 增至 295，解码为 38.51 对 38.47 token/s，第 319 个 token 首次不同。测试热状态 nominal，峰值 42.25 GiB，系统可用内存最低 51%，pageout 和 swap 计数不变。线上继续使用原宽度策略。见 `runs/narrow-draft-interleaved.log`。

从同一份 4K 缓存实测线性验证 6、7、8、9、11 个位置。当前路径中位数分别为 75.30、77.92、78.53、92.27 和 92.27 ms。为 7、8、9 个位置启用编译 DeltaNet 后为 76.34、76.90 和 90.67 ms。每个宽度的编译及未编译隐藏状态、预测、提交后 KV 和 FP32 状态一致。实验峰值 41.97 GiB，系统可用内存最低 51%。见 `runs/verification-width-breakpoint.log`。

原 4K/1K 策略的 266 轮只有 12 轮是 7、8 或 9 位置且无分叉，分别有 3、3、6 轮。首次交替测试中基线总是先执行，承担了共享图冷启动，测得 38.48 对 38.86 token/s。预热两组宽度并轮换先后顺序后，基线与扩展编译分别为 38.80 和 38.77 token/s，输出逐 token 一致。两路共同峰值 42.26 GiB，最低可用内存 51%。线上保留 6 和 11 两种编译宽度。按实测宽度成本及每次草拟平均 2.11 ms，从 8 扩到 11 位置约多付 20.1 ms，预计每轮确认数至少需增加约 22% 才值得扩宽。见 `runs/extra-compiled-widths-interleaved.log` 和 `runs/extra-compiled-widths-balanced.log`。

草拟到第 7 步后，用前七步概率的几何平均估计后续概率，并仅在预期新增确认数覆盖 8 → 11 的实测耗时时继续扩宽。根部分叉保留在一个 8 位置块内。4K/1K 同模型交替对照中，超过 8 位置的验证从 113 降至 78 轮，单轮主模型平均耗时从 83.17 降至 80.86 ms。总验证轮数由 266 增至 288，解码从 38.94 降至 37.25 token/s，第 319 个输出 token 首次不同。峰值 42.25 GiB，系统可用内存最低 51%，服务恢复后通过健康检查。见 `runs/cost-adaptive-panels-interleaved.log`。

在原策略上采样实际草拟概率，复现相同输出和 266 轮。99 轮产出 10 个主链候选，其中 92 轮无分叉。按上述规则值得扩宽的 76 轮，预测前七步全中率 82.05%，实测 31.58%，预计额外接受 2.34 个 token，实测 0.55 个。相比同一轮截在七个候选，实际确认数增加 10.14%，低于约 22% 的成本比例。七步概率至少 95% 的 22 轮也只命中 13 轮，平均实际确认数增幅 11.43%。这是单输入、固定轮起点的比较，真实截断会改变后续轮起点。本轮不采用该扩宽规则。采样峰值 41.73 GiB，系统可用内存最低 51%。见 `runs/panel-gain-calibration.log` 和 `runs/panel-gain-calibration.json`。

用四层实际 BF16 gate/up 权重试验矩阵乘法操作数的精确重编码。权重和激活分别乘以 256 和 8，临时值可精确表示时使用 FP16 操作数，否则用 FP32，FP32 部分和恢复原比例后再做 SwiGLU。6、11、16 位置的 BF16 SwiGLU 输出逐位一致。每块矩阵内做往返核对时，11 位置从 0.600 增至 1.073 ms。改为用现有权重包指数描述符提前判断，约 96.75%–97.25% 的权重包满足条件，激活转换和检查也移出乘加热循环。计入激活预处理，6 位置为 0.480 → 0.507 ms，11 位置为 0.595 → 0.631 ms，16 位置为 0.581 → 0.639 ms。随机 BF16 激活和含很小数值、会触发 FP32 回退的合成激活通过逐位比对。微基准峰值 4.13 GiB，系统可用内存最低 71%，线上恢复后仍为原版 BF16 配置。见 `runs/scaled-half-gate-up.log` 与 `runs/preclassified-half-gate-up.log`。

四层 FFN down 试验三位指数与逐 lane 原样回退。每个权重包用位图查表和 popcount 定位两路数据，去掉逐权重的 SIMD 前缀扫描。12.18%–14.03% 的 lane 保存完整 BF16，编码载荷由原始大小的 76.15%–76.20% 降至 74.12%–74.70%。全量还原及 6/11/16 位置的 FP32 部分和逐位一致。投影耗时分别为 0.264 → 0.268、0.356 → 0.366、0.356 → 0.370 ms。另测保留 BF16 低字节只压缩高字节，载荷为 76.41%–76.56%，11 位置为 0.356 → 0.358 ms，同样逐位一致。两项峰值最高 2.88 GiB，系统可用内存最低 71%，实验热状态 nominal。线上保留原布局并通过健康检查。见 `runs/bf16-lane-escape-probe.log` 与 `runs/bf16-low-byte-probe.log`。

三位指数保持逐 lane 无损回退，试验向量化还原四对 BF16，以及把三个指数存储字节压入连续的对齐字流。11 位置四位基线、三位原版、仅向量还原、合并读取分别为 0.351、0.365、0.361、0.385 ms。6、11、16 位置的 FP32 部分和均逐位一致，热状态 nominal，峰值 2.96 GiB，系统可用内存最低 71%。见 `runs/bf16-lane-word-probe.log`。

每个张量从 64 行各采样 1024 个原版 BF16 权重。FFN gate/up/down 的独立标量熵约为 10.48 位，指数约 2.56–2.57 位，符号和尾数的条件熵约 7.91 位。同一行的前一个指数对条件熵的影响小于 0.002 位。独立标量熵对应原字节数约 65.5%，这个样本估算没有计入 GPU 布局、索引与解码成本。见 `runs/bf16-scalar-entropy-audit.log`。

用同四层已编码 down 权重测读取、还原和 11 位置投影，共 543,176,448 字节。三种内核分别为 1.000、0.935 和 1.441 ms，对应每秒处理 543、581、377 GB 的已编码权重。读取和还原分别通过模 2^32 校验和，使用不同的调度及小输出，时差不能直接当作算子各阶段耗时。完整投影的有效权重吞吐低于两个只读对照，继续优化还原布局或乘加的调度仍有实测空间。峰值 1.46 GiB，系统可用内存最低 72%。线上 BF16 服务恢复后通过健康检查。见 `runs/down-read-decode-cost.log`。

按线程组读取顺序交错重排无损权重包，描述符保留逻辑位置。四层 down 和四层 gate/up 全量权重还原逐位一致，6、11、16 位置的 FP32 部分和或 BF16 SwiGLU 输出同样一致。11 位置的 down 为 0.3534 → 0.3524 ms，gate/up 为 0.5966 → 0.5949 ms。布局字节数相同，微基准峰值 5.76 GiB，系统可用内存最低 70%。线上保留原排布。见 `runs/bf16-packet-schedule-probe.log`。

从同一份 4K 缓存交替用 6、8、11 和 16 位置跑完整目标验证，每种宽度都喂入相同的 1023 个已记录 token。单轮中位数分别为 74.52、77.09、91.68 和 97.81 ms。每种宽度的 1018 个预测位置匹配记录，首处差异均在第 430 个输出 token。把实测验证中位数加上每次草拟打分 2.11 ms 的估算，11 位置在全接受时为 97.53 token/s，16 位置到 100 token/s 需平均接受 12.95 个位置。该回放强制使用已记录输入，速率是条件估算，未计入独立回滚或未接受分叉的额外成本。峰值 42.67 GiB，系统可用内存最低 51%，换页和交换计数不变。见 `runs/target-teacher-roofline.log`。

24、32 位置的无损 FFN 内核复用同一次权重读取，测试的 FP32 down 部分和和 BF16 SwiGLU 输出逐位对应分块现有内核。同一 SIMD 组持有 3/4 个 token 块时，down 分别为 0.935/1.192 ms，gate/up 为 1.704/2.257 ms，均慢于分块调用。将 BF16 down 权重先还原到线程组共享内存，再让每个 SIMD 组独立计算一块后，24/32 位置为 0.591/0.667 ms，接近现有的 16+8 或 16+16 分块调用（0.545/0.649 ms）。微基准峰值最高 4.13 GiB，系统可用内存最低 71%。线上保留原内核并通过健康检查。见 `runs/wide-lossless-panels.log` 和 `runs/shared-weight-tile-down.log`。

使用线上共享的 32K 草拟词表，在真实 4K/1K 输出的 1023 个起点用已确认 token 复放 MTP，最多 31 步。连续 16 步全为首选的起点有 16 个，连续 23 步只有 1 个。允许正确 token 为前两名时，这两个计数为 39 和 9。预知正确路径并按验证节点预算计入次优候选，16/24/32 个验证位置的最少轮数分别为 263/261/260（首选）、194/190/187（前两名）、175/168/166（前四名）。假设宽于 16 的目标验证仍等同 16 位置实测成本，且未命中分支不增加开销，最宽的前两名和前四名方案分别估算为 61.47 和 66.16 token/s。限制 16 位置、最多两处分叉、只允许分差不超过 2.0 的第二候选时为 223 轮及 53.99 token/s 的乐观估算。这些数字使用已知输出衡量覆盖与宽度收益，在线规划仍需要识别正确分支。同一次基线生成 38.62 token/s、266 轮，文本哈希与线上记录一致。整次单模型峰值 42.75 GiB，草拟排位阶段峰值 40.63 GiB，服务恢复后通过健康检查。见 `runs/deep-mtp-oracle.log`、`runs/deep-mtp-oracle-trace.json` 和 `runs/deep-mtp-two-fork-budget.log`。

将上轮主模型所有未接受节点的预测按绝对位置对齐下一轮，检查复用为额外分叉候选的机会。插桩复测生成相同 1024 个 token 和 266 轮，实测解码 38.60 token/s。在 242 次根部有预测的轮次中，最多票预测与 MTP 首选不同的 185 次里，能补中 3 次，另有 139 次原 MTP 已正确。对已接受路径的末端，167 轮有可对齐的预测且能容纳一个节点，其中 17 轮的候选集合含正确 token，最多票候选可在 15 轮补中。固定在第二草拟位置加一个不同的备选并保持 8/16 位置档位，历史上会触发 154 轮，包含 136 轮新建树并最多命中 5 轮，原有树的 18 轮均未命中。这是固定原调度的机会计数，未运行该增设分支策略。测试峰值 42.75 GiB，系统可用内存最低 51%，服务恢复后通过健康检查。见 `runs/discarded-prediction-bank.log` 与 `runs/discarded-prediction-bank.json`。

在无损 BF16 草拟词表投影的写回位置合并局部前两名和归一化统计，再与提示词额外词表项一次归并。真权重和合成激活上的第一名、第二名、BF16 分差及跨块并列顺序一致，log 概率最大绝对误差 4.77×10⁻⁷。32K 词表带 768 个额外项时，单次完整的两名排序从 0.495 降至 0.417 ms。4K/1K 按轮交替对照启用了 1794 次融合路径，生成 1024 个相同的 token，均验证 266 轮。解码 38.75 对 38.78 token/s，草拟 4.053 对 4.038 s，目前策略在多数位置只需第一名。整模两路峰值 42.26 GiB，系统可用内存最低 51%，热状态 nominal。线上使用原内核并通过健康检查。见 `runs/fused-lossless-draft-rank.log` 和 `runs/fused-rank-interleaved-ab.log`。

保留原主链和根分叉，用融合排序在第 2–7 个草拟位置选一个最小分差、且分差不超过 2.0 的第二候选。仅在已有树或已有宽线性轮上加入一个节点，宽度保持在原 8/16 位置档位。已有侧链继续读取原长度的掩码，新增节点按它的实际祖先单独计算注意力。4K/1K 按轮交替对照逐 token 一致。新增节点触发 76 次、接受 6 次，树轮数从 57 增至 114，但两组验证总数均为 266。目标耗时从 22.10 增至 22.49 s，解码从 38.98 降至 38.44 token/s。两路只驻留一份模型，峰值 42.32 GiB，系统可用内存最低 51%；热状态 nominal，pageout 和 swap 计数保持不变。试验结束后恢复线上原策略并通过健康检查。见 `runs/same-tile-late-fork.log`。

DeltaNet 独立回滚中，每个线程处理相邻八个 FP32 状态值。合成实际形状的线性和索引路径均与原恢复逐位一致，8 步带索引从 0.0374 降至 0.0242 ms。原 4K/1K 按轮交替，两路各回滚 11,712 个层状态，1024 个输出 token 逐个一致，均验证 266 轮，解码均约 38.98 token/s。整模峰值 42.26 GiB，系统可用内存最低 51%。见 `runs/vector-delta-restore.log` 和 `runs/vector-restore-interleaved.log`。

将每六层的回滚合并为一个 Metal 调度，直接读取各层缓冲，输出各层状态切片。24 层合成样本的 8 步索引回滚从 0.816 ms 降到 0.447 ms，输出逐位一致。整模 4K/1K 对照中，合并路径构造 1,952 组，覆盖原 11,712 个层状态，1024 个输出一致，均验证 266 轮。解码从 38.93 至 38.86 token/s，整段目标验证约 22.15/22.19 s，两路峰值 42.28 GiB，系统可用内存最低 51%。热状态 nominal，pageout 与 swap 计数不变。线上保留原回滚路径。见 `runs/grouped-delta-restore.log` 和 `runs/grouped-restore-interleaved.log`。

将 4K 提示词中最常见的 512 个 token 的 MTP 输入嵌入投影贡献缓存为 FP32，请求内表和索引共 10.13 MiB，准备耗时 23.56 ms。命中时 Metal 只读取 MTP 投影的隐藏状态半矩阵，未命中在设备上用原 BF16 权重完成两部分乘加。投影微基准命中为 0.123 → 0.066 ms，未命中为 0.125 → 0.161 ms。两条路径的 FP32 累加顺序与原实现不同，微基准中的 BF16 输出分别有约 0.044% 和 0.015% 的元素变化。初次微基准漏做了与生产路径相同的 BF16 转回，以下结果为修正后的复测。

4K/1K 同模型按轮交替对照中，实际命中 1,218/1,657 次（73.51%），草拟耗时从 4.024 降到 3.993 s，验证轮数由 266 增至 267。两路 1024 个输出 token 逐个一致，解码为 38.90 → 38.73 token/s，计时从共享预填充后的解码开始。解码共同峰值 42.27 GiB，系统可用内存最低 51%，热状态为 nominal，pageout 和 swap 计数不变。线上保留原配置，恢复后 BF16 HTTP 健康检查通过。见 `runs/sparse-mtp-projection-retest.log`。

短批全注意力的 Q/K 路径把 RMSNorm、BF16 舍入、部分 RoPE 和布局变换合为一次 Metal 调度。四组原版归一化权重使用 BF16 合成投影，6、11、13 个位置的线性和树形 Q/K 均逐位一致。11 位置线性耗时为 0.088 → 0.057 ms。输出侧把 BF16 Sigmoid 门控、乘法和投影片段重排合为一次 Metal 调度，6、11、16 位置和四档门控输入尺度通过逐位对照，11 位置为 0.067 → 0.056 ms。首版门控使用 FP32 Sigmoid，复测采用与 MLX BF16 操作一致的中间舍入。见 `runs/fused-qk-norm-rope-retest.log` 和 `runs/fused-attention-gate-pack-retest.log`。 [MLX RMSNorm](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/rms_norm.metal)、[MLX RoPE](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/rope.metal)、[MLX Sigmoid](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/unary_ops.h)

共用同一份模型和 4K 预填充，6、8、11 位置线性及 12 节点树的主模型隐藏状态、K 缓存和预测均一致。按轮交替生成同一 1K 输出，单用 Q/K 融合时解码为 38.91 → 39.06 token/s，两项合用为 38.80 → 39.01 token/s，两组均为 266 轮且输出逐 token 一致。合用对照的解码共同峰值 42.40 GiB，系统可用内存最低 51%，pageout 和 swap 计数不变。热状态至末次采样由 nominal 转为 fair，四层固定 FFN 耗时 3.37–3.49 ms。见 `runs/fused-qk-interleaved.log` 和 `runs/fused-attention-edges-interleaved.log`。

短批注意力融合已启用于 M2 Ultra、MLX 0.32.4 和当前 BF16 头尺寸的优化 HTTP 配置。正常 HTTP 入口的 4096/1024 对照从 38.67 升至 38.93 token/s，端到端从 42.79 降至 42.47 s，两组文本哈希一致且均验证 266 轮。新版请求峰值 42.75 GiB，解码峰值 41.32 GiB，系统可用内存最低 50%。两次 HTTP 采样前后热状态均为 nominal。短请求、模型加载互斥、超上下文拒绝和请求后健康检查通过。首次上线验证把 262,000 token 上限误当作固定会超工作集预算的探针。服务已正常完成该请求，后续复测沿用同一基线，通过进程锁和上下文上限分别核对加载与请求准入。见 `runs/http-fused-attention-io.log` 和 `runs/http-fused-attention-io-retest.log`。

用当前线上配置截取 266 轮真实 MTP 草拟的隐藏状态，在主链首处拒绝之前逐位置复放。补扫复用完整 BF16 词表编码中的 215,552 个尾部行，不加载额外权重，也不重投影固定的 32K 行。266 个根位置有 59 次拒绝，其中 4 次正确 token 不在请求草拟词表，尾部最高分取回其中 3 个。975 个有效草拟位置（含 244 次首处拒绝）中，有 26 次正确 token 不在词表，尾部最高分取回其中 18 个。取回表示成为该补集首选，部分候选分数仍低于原主链候选。单位置补扫中位耗时 2.53 ms，审计阶段峰值 40.57 GiB，整体请求峰值 42.75 GiB，系统可用内存最低 51%。基线完整复现 1024 个输出 token 和 266 轮。见 `runs/mtp-complement-audit.log` 与 `runs/mtp-complement-audit.json`。

补集分支对照仅在可增加一个同 8/16 位置档位节点的轮次考虑补扫。最近 16 个已确认 token 中出现过原草拟词表外 token 时，批量评估该轮主链已草拟位置；其余轮次最多评估一个置信度最低且 log 概率不高于 -0.6 的位置。选中词表外 token 至少领先现有候选 1.0 分时才加一个备选节点。4K/1K 单模型按轮交替对照中，补扫 162 次、合计 653 个位置，加入 17 个节点并接受 8 个。输出 1024 个 token 逐个一致，验证 266 → 265 轮，目标耗时 22.16 → 22.10 s，草拟耗时 4.07 → 4.54 s，解码 38.81 → 38.23 token/s。共同解码峰值 42.27 GiB，系统可用内存最低 51%，热状态 nominal，pageout 和 swap 计数不变。线上沿用现有草拟策略，BF16 HTTP 服务已恢复并通过健康检查。见 `runs/mtp-complement-branch-ab.log`。

用真实 4K/1K 草拟轨迹复现 975 个有效查询位置，在每轮开始前已经确认的最近 512 个有效位置中按余弦相似度检索。512 行原样 BF16 状态共 5.00 MiB，每条记录关联上一个 token 和已确认的续写位置。限定前一个 token 相同，存在近邻的 685 个位置中，直接替换首选会修正 22 次错失、破坏 197 次正确草拟。限定在同一 8/16 位置档位内最多加一条续写链，余弦阈值为 0.65，固定轨迹上有 53 轮可补分支，首 token 命中 7 轮，连续 2/3 token 分别命中 6/3 轮，估算额外接受 15 个 token。这段离线回放仅比较首处拒绝之前的查询，实际调度仍可能选到已偏离的草拟前缀。记录轨迹时的整段峰值为 42.75 GiB，系统可用内存最低 51%。见 `runs/mtp-neighbour-audit.log` 与 `runs/mtp-neighbour-audit.json`。

4K/1K 交替对照将近邻查询接入实际草拟，使用 512 行原样 BF16 环形缓存，仅写入主模型已确认前缀对应的状态，最多加一条长度为 3 的同档位续写链。221 次搜索加入 79 条链、204 个候选节点，4 条链获选，接受其中 7 个 token。两组 1024 个输出 token 逐个一致，均验证 266 轮。树轮数从 57 增至 129，目标耗时从 22.03 增至 22.61 s，草拟从 4.05 增至 4.44 s，解码从 39.05 降至 37.48 token/s。共同解码峰值 42.33 GiB，系统可用内存最低 51%，热状态为 nominal，pageout 和 swap 计数不变。线上使用原草拟策略并通过 BF16 HTTP 健康检查。见 `runs/mtp-neighbour-branch-ab.log`。

FFN down 写回试验保留 34 个 FP32 部分和的原归约顺序，合并 BF16 舍入及残差相加。分别测试原部分和排布和按八个输出特征集中排布；四层原版权重在 2、6、8、11、16 位置的 FP32 部分和及最终 BF16 残差均逐位一致。6 位置原路径、原布局融合、新布局融合分别为 0.2656、0.2676、0.2655 ms，11 位置分别为 0.3566、0.3656、0.3627 ms。微基准峰值 1.91 GiB，系统可用内存最低 72%，保留原写回。见 `runs/ffn-tiled-writeback.log`。

让相邻两个 FFN down 输出块共用激活片段，直接使用目前的独立无损权重包。四层真实矩阵的 6、11、16 位置均逐位匹配原 FP32 部分和。11 位置原内核、跨半区配对、相邻配对分别为 0.3104、0.2986、0.2954 ms；6 位置为 0.2596、0.2567、0.2568 ms。选择相邻配对做整模对照。两路复用所有权重缓冲，各有请求状态和对应的编译图。6、8、11 位置的整模隐藏状态和预测逐位一致，4K/1K 输出逐 token 一致并均验证 266 轮。解码为 39.09 → 38.67 token/s，目标验证耗时 22.01 → 22.32 s。共同解码峰值 42.26 GiB，系统可用内存最低 51%，热状态 nominal，pageout 和 swap 计数不变。此整模型结论覆盖相邻配对。线上保留现有内核，服务恢复后通过 BF16 健康检查。见 `runs/independent-down-current-ab.log`。

DeltaNet 延迟提交把前一轮确认路径的 FP32 状态恢复放进下一轮验证的递推内核。每层保留一份稠密起点及固定 16 位置的秩一更新、归一化 K 和衰减记录，下一轮在寄存器中重放已确认路径，再加载新候选。用真实形状的合成输入覆盖线性、树和双分叉，BF16 输出、归一化 K、FP32 更新以及连续两轮的提交状态逐位一致。含独立恢复的合成对照中，树转 6 位置线性由 0.257 降至 0.235 ms，6 转 11 线性由 0.279 降至 0.265 ms。微基准峰值 0.037 GiB。见 `runs/deferred-delta-recurrence.log`。

整模对照先用 6、13、11、2 位置串行复放，对照了实际线性保留、整段提交及树路径 [0,11,12]，隐藏状态、预测、卷积历史、KV 及物化后的 FP32 状态逐位一致。按轮交替生成 4K/1K 时，候选延迟 10,032 个线性层状态和 2,736 个树形层状态，两路均验证 266 轮，1024 个输出 token 逐个一致。目标累计耗时 21.98 → 22.40 s，解码 39.18 → 38.56 token/s。共同峰值 42.26 GiB，系统可用内存最低 51%，热状态 nominal，pageout 和 swap 计数不变。线上保留原提交路径。见 `runs/deferred-delta-interleaved.log`。

另测将 DeltaNet Q/K 归一化从逐值通道递推移到单独的按头内核。合成实际形状的 6、11 位置线性和 13 节点树输出及状态逐位一致，计入新增调度后分别从 0.216、0.210、0.248 ms 增至 0.230、0.227、0.253 ms。微基准峰值 0.017 GiB，系统可用内存最低 74%。服务恢复后通过健康检查。见 `runs/delta-shared-qk-norm.log`。

一个 SIMD 组改为同时更新同一 value head 的两个相邻状态行，两行共用寄存器内归一化后的 Q/K 和衰减。合成实际形状的 6、11 位置线性及 13 节点单分叉、双分叉的 BF16 输出、FP32 状态、秩一更新和回滚逐位一致。11 位置递推由 0.236 降至 0.226 ms，13 节点单分叉由 0.253 降至 0.239 ms。整模用同一份权重、分别编译后，6、13、11、2 位置串行复放的提交状态也逐位一致。4K/1K 按轮交替对照的 1024 个输出 token 均一致，验证均为 266 轮，目标耗时从 22.00 增至 22.27 s，解码从 39.15 降至 38.73 token/s。解码共同峰值 42.26 GiB，系统可用内存最低 51%，热状态 nominal，pageout 和 swap 计数不变。见 `runs/paired-delta-rows.log` 与 `runs/paired-delta-interleaved.log`。

用 FP32 交错布局存储成对状态，让递推写回和回滚读取使用向量加载和存储。合成实际形状的输出及恢复到原布局的状态逐位一致。含回滚微基准中，原路径、双行原布局、双行交错状态在 6 位置分别为 0.258、0.249、0.243 ms，11 位置为 0.267、0.255、0.252 ms，13 节点树为 0.283、0.268、0.258 ms。峰值 0.038 GiB，系统可用内存最低 74%。交错布局只测了内核，计时未含一次性状态转换。线上保留原递推及状态布局，服务已恢复并通过 BF16 健康检查。见 `runs/delta-interleaved-state.log`。

把 6/8/11 位置的实测目标成本及每次草拟 2.11 ms 加入真前缀排行回放。若提前知道每个起点的正确草拟长度，再用动态规划按新的确认边界选择宽度，固定 6、8、11 的估算分别为 38.54、38.94、32.98 token/s，可在三档中任选为 42.74 token/s。92 个原宽线性轮中，有 25 轮首 7 个草拟均正确，15 轮的本轮接受增量覆盖额外成本，计入理想后续轮次后只有 8 轮仍显示收益。用四段互相留出验证，从七步累计置信度、根置信度、根前两名分差、上一轮及前四轮平均确认数学习单阈值，各规则在留出段最多选一轮扩宽且未获得正收益。数据来自同一 4K/1K 轨迹，离线 DP 假定已知后续匹配并只允许三种线性宽度。它不能代表有侧链或更宽批次的实时吞吐。另在 85 个线性 5 候选轮中只有 6 轮全部命中，其中用真前缀排行向后补至 7 只在 2 轮找到额外命中。这些结果不支持再调现有扩宽门槛以接近 50。见 `runs/verify-width-causality.log`。

在 M2 Ultra 上用 Instruments 采集同一份已预热 4K 缓存的 6/11 位置线性验证。只对该 Python 进程的 Compute Active 区间取并集，以连续 195 个区间识别完整样本，分别得到 41 和 32 轮。首末 GPU 区间的中位跨度为 72.00 / 88.47 ms，GPU 活跃为 71.78 / 88.25 ms，内部空档仅为 0.20 / 0.22 ms。宽度切换新增的 16.47 ms 基本都在 GPU 执行区间。这里的活跃比例是时间覆盖率，不是 SIMD 占用率。原基准为每轮克隆同一份 4K KV 起点，重复包含跨 4096 边界的缓存扩容，且不计轮间空档、草拟或回滚。当前追踪导出了编码器区间，未取得逐着色器耗时，不能据此把增量归到某个具体内核。所有验证预测与记录一致，追踪期间峰值 41.47 GiB，系统可用内存最低 51%，服务已恢复且 BF16 健康检查通过。见 `runs/verification-metal-trace.log` 和 `runs/verification-metal-timing.log`。

主模型全注意力的短批 Q/K 预处理内核直接输出合并的 BF16 K/V 缓冲，追加时只更新一次。非连续树路径只记录提交索引，在下一轮与新位置一起追加。6、13、11、2 位置的串行复放包含容量扩展和树路径 [0,11,12]，隐藏状态、预测、已提交 K/V、卷积历史和 FP32 DeltaNet 状态逐位一致。4K/1K 单模型交替实测执行 4,256 次合并追加和 256 次延后分支提交，1024 个输出 token 逐个一致，两路均为 266 轮。解码 39.14 → 39.15 token/s，目标耗时 21.97 → 22.03 s，提交耗时 0.125 → 0.076 s。共同解码峰值 42.26 GiB，系统可用内存最低 51%，热状态 nominal，计时期间 pageout 和 swap 计数不变。整段耗时差约 8 ms，线上保留独立 K/V，BF16 服务恢复后通过健康检查。见 `runs/fused-main-kv-interleaved.log`。

短批全注意力在同一 SIMD 组内让两个相邻查询共享 K/V 读取，同时保留 MLX 的 128 个循环分区、BF16 部分和及最终 FP32 归约顺序。合成 BF16 输入覆盖 6–13 个查询、4102–5101 个 KV 位置，128 分区全部逐位一致，64 分区不一致。6 位置微基准从 0.460 降至 0.437 ms，11 位置在 4107/5017 个 KV 时从 0.637/0.710 降至 0.587/0.650 ms，峰值 0.059 GiB。整模串行复放验证了线性及非连续树路径的已提交状态，4K/1K 单模型交替对照从 38.82 升至 39.07 token/s，目标验证耗时从 22.16 降至 21.97 s，1024 个输出逐 token 一致且均为 266 轮。共同解码峰值 42.26 GiB，系统可用内存最低 51%，热状态 nominal，计时期间 pageout 与 swap 计数不变。见 `runs/exact-shared-gqa.log` 和 `runs/exact-shared-gqa-interleaved.log`。 [MLX 分区选择](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp#L685) [MLX 两阶段归约](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/sdpa_vector.h#L168)

共享 K/V 读取的候选也经过正常 HTTP 入口复测。首次候选从 nominal 转为 fair，解码 13.17 token/s，同场对照 38.86。随后同一驻留服务的基线、候选、基线分别为 19.70、19.09、17.62 token/s，固定 FFN 探针随时间同步变慢。最终在 nominal 下复测，候选为 37.88 token/s，对照为 38.86，输出哈希一致且均为 266 轮，峰值 42.75 GiB，系统可用内存最低 50%。候选未上线，BF16 服务恢复并通过健康检查。见 `runs/http-shared-kv-attention.log`、`runs/shared-kv-resident-http.log` 和 `runs/http-shared-kv-attention-retest.log`。

在 8 GiB 预算内用四层原版权重测试相邻两个 FFN gate/up 输出块共用激活片段。6 和 11 位置的 SwiGLU 结果逐位一致，单层耗时分别由 0.496 增至 0.901 ms、由 0.606 增至 1.370 ms。微基准峰值 4.13 GiB，系统可用内存最低 71%，热状态 nominal。线上保留原 FFN 内核并通过 BF16 健康检查。见 `runs/ffn-projection-tile-pair.log`。

请求准入用预计 KV 和临时缓冲检查当下的统一内存余量。八个主机采样用例通过，其中包含缓存释放后恢复、重新采样失败和 Metal 预算先行拒绝。模拟可用内存为 15% 时，正常 HTTP 处理器在预填充前返回 503，试验未制造真实内存压力。部署重启后短请求和互斥锁检查通过，4096/1024 HTTP 实测解码 38.08 token/s，端到端 43.01 s，输出哈希一致且验证 266 轮。峰值 42.75 GiB，系统可用内存最低 50%，热状态 nominal，请求后健康检查通过。见 `runs/unified-memory-admission-cases.log` 和 `runs/http-unified-memory-headroom.log`。

在 8 GiB 预算内用同四层 gate/up 测读取、BF16 还原和完整投影，共处理 1,091,459,584 字节编码权重。四层纯读取、还原与校验、6 位置投影、11 位置投影分别为 1.726、1.715、2.020、2.398 ms，有效权重吞吐分别为 632、636、540、455 GB/s。权重往返逐位一致，读取和解码校验和相同。纯读取使用单独的流式调度，不能把测得的时差直接当作可节省的时间。本对照显示还原路径已经达到纯读的吞吐量级，较宽批次的投影仍有额外成本。峰值 3.22 GiB，系统可用内存最低 71%，热状态 nominal，试验后 BF16 HTTP 服务恢复并通过健康检查。见 `runs/gate-up-cost-breakdown.log`。

在现行 4K/1K 的 266 轮验证中复放前五个草拟位置。只在原 8/16 档位至少余两个位置、且该轮没有分支时，比较第一、第二候选各自续写一步的置信度。211 个备选检查覆盖 139 轮。按分差最小选时补中 16 轮、估计接受 25 个候选；加入下一步置信度后补中 15 轮、估计接受 23 个候选。要求置信度优势抵消原分差后只补中 1 轮。这些计数来自固定轨迹，分支策略没有接入在线生成。捕获的隐藏状态约 9.98 MiB，整模峰值 42.75 GiB，复放阶段峰值 40.67 GiB。见 `runs/mtp-counterfactual-lookahead.log`。

紧凑树注意力按逻辑祖先位置分配原 128 个循环分区，直接读取物理 KV 槽。两个查询共享前缀读，尾部按各自祖先索引读取。两种线性宽度和三种树形在完整输出上逐位匹配逐查询拼接祖先的参考。现行树掩码在这三种树形有 1,697–2,287 个 BF16 元素与参考不同。8 节点树由 0.593 降到 0.488 ms，两种 13 节点树由 0.817/0.873 降到 0.668/0.720 ms。合成试验峰值 0.55 GiB，系统可用内存最低 71%，服务恢复后通过健康检查。见 `runs/ancestor-cyclic-attention.log`。

整模把同三个 token 放入 13 位置的线性路径和非连续树路径 [0,11,12]。新注意力下，已确认位置的隐藏状态与预测、提交的 KV/卷积/DeltaNet 状态，以及下一轮线性隐藏状态与预测均逐位一致。同一 4K/1K 缓存的按轮对照中，现行路径和候选分别为 37.84、37.79 token/s，验证 266、267 轮，首个输出差异为第 430 个 token。共同解码峰值 42.26 GiB，系统可用内存最低 51%，热状态 nominal，测量期间 pageout 与 swap 计数不变。线上保留现有内核。见 `runs/ancestor-attention-interleaved.log`。

第 430 个输出的两条路径有相同的前 429 个 token，且都在起点 426 的 11 位置线性验证中打分。现行路径对 token 1439 和 411 的 BF16 得分为 22.625、22.375；新祖先布局为 22.375、22.375，按较小 ID 的并列规则选中 411。从同一 4K 缓存重放相同的 429 个 token，分三组用单 token 前向处理最后 16 或 48 个位置。完整词表打分三组均选中 1439。此处证据支持现行结果，祖先布局保留在试验中。复核峰值 53.93 GiB，系统可用内存最低 51%。见 `runs/ancestor-attention-first-difference.log`。

树形候选改用现行掩码的物理 KV 位置分区，并用每个查询的位图选择尾部祖先。8/13/13 节点三组输出均逐位匹配现行内核，耗时由 0.599/0.824/0.880 降至 0.495/0.672/0.721 ms。整模先通过 6、13、11、2 位置的隐藏状态、预测与缓存逐位检查，再在同份模型上交替执行 4K/1K。两组 1024 个输出逐 token 一致，均为 266 轮，特殊路径执行 656 次。累计验证耗时由 55.12 降至 54.98 s，解码为 16.58 → 16.61 token/s，约 0.22%。计时从 nominal 转为 fair，固定四层 FFN 探针为 5.67–9.36 ms。共同峰值 42.26 GiB，系统可用内存最低 51%，pageout 与 swap 计数不变。该样本的整段收益不足以启用候选，线上保留原内核。见 `runs/physical-tree-attention.log` 和 `runs/physical-tree-attention-ab.log`。

从原 4K/1K 的 92 个 11 位置线性轮捕获第 24、40、56 层的 BF16 隐藏状态，共 18.87 MiB。用原权重短名单预测宽验证收益。实测单批后续层从 11 缩至 8 位置时，保留位置的隐藏状态和预测逐位一致。92 轮中有 18 轮会因截到 8 位置丢掉已确认候选。第 24 层即使预知安全截断位置，扣除预览耗时后的估算最多节约 0.617 s，对应 39.80 token/s，未计分段开销。跨四段留出的最大分差规则截窄 9 轮，其中一轮丢掉两个正确候选。没有接入在线调度。整次模型加载和试验峰值 53.93 GiB，系统可用内存最低 51%，热状态 nominal，测量期间无新增换页或交换。见 `runs/late-layer-width-audit.log`。

用同一请求的已确认历史对比严格后缀、允许一次替换，以及把替换映射应用到候选续写。严格后缀在 12 个轮首共匹配 73 个候选，这些轮的现行验证确认 54 个候选。允许一次替换的长续写只在 5 个轮首触发，匹配 18 个，现行为 26 个。把替换应用到续写后，3 个轮首匹配 3 个，现行为 10 个。两种允许替换的方法在已留空位的分支上都没有补中。全轨迹查询耗时约 6 ms，仅使用已出现的 token，分支统计以固定轨迹估算。见 `runs/fuzzy-suffix-causal-audit.log`。

在线严格复制采用至少 10 个连续匹配 token，要求源续写与当前匹配段不重叠，历史索引最多覆盖 8192 token。4K/1K 单模型交替对照中触发 11 轮，复制候选确认 48/110，草拟打分 1970 → 1928 次，验证 266 → 270 轮。解码为 38.35 → 37.83 token/s，第 763 个输出首次发生变化。共同峰值 42.26 GiB，系统可用内存最低 51%，热状态 nominal，换页和交换计数不变。现行服务保留原策略。见 `runs/anchored-exact-lookup-ab.log`。

用提示词的 4094 个已知后继标注原 MTP 的首选和第二候选。校准耗时 0.94 s，暂存原样 BF16 隐藏状态 40 MiB。用两种条件统计已有修正，一种按前一 token 和首选共同分组，另一种只按首选分组。目标输出的 810 个有效草拟位置中，第二候选正确 68 次，其中 42 次分差不超过 2.0。要求提示词中相同修正至少出现两次、且频次至少为保留首选的两倍时，两种条件均未提出可进入当前验证档位的额外分支。整段基线复现相同输出哈希和 266 轮，加载和试验峰值 53.93 GiB，系统可用内存最低 51%，结束后 BF16 HTTP 服务恢复。见 `runs/prompt-calibrated-mtp-audit.log`。

用四层原版权重把 gate/up 的 FP16 操作数判断提到整个 K 分片外，热循环直接从指数和尾数生成按二的幂放大的操作数，累加仍为 FP32。只对能精确表示的分片启用，其余使用原 BF16 还原及乘加。约 58.8%–64.0% 的权重分片满足条件，正常和触发回退的输入均逐位匹配 6/11/16 位置的现行 SwiGLU。11 位置的内核由 0.605 增至 0.752 ms，此外激活检查和转换约 0.290 ms。峰值 4.12 GiB，系统可用内存最低 71%，线上保留原投影。见 `runs/whole-split-exact-half.log`。

保留 512-token 的注意力和 DeltaNet 分块后，将同层多个完整块的逐行 FFN 合批。三层原版权重的 4096 位置合批从每层 131.51 降至 119.82 ms，输出逐位一致，微基准峰值 2.49 GiB。单模型 4K 整段验证中，现行块式、按层、按层合批 FFN 的首 token 分别为 16.18、13.75、13.42 s，全部预填充隐藏状态、KV/卷积/DeltaNet 和 MTP 缓存逐位一致。两路交替解码均生成相同的 1024 个 token，验证 266 轮，端到端从 42.84 降至 39.89 s。合批预填充峰值 42.77 GiB，共同解码峰值 41.89 GiB，系统可用内存最低 51%，热状态 nominal，换页和交换计数不变。见 `runs/prefill-ffn-panels.log` 和 `runs/prefill-panel-ffn-ab.log`。

正式路径把合批限于完整的 512-token 块，余块走原路径。首次把 17-token 余块合入时逐位检查失败；调整后 520 和 1041 token 两种分块的隐藏状态和所有缓存逐位一致。同场正常 HTTP 入口的 4096/1024 实测首 token 16.19 → 13.55 s，端到端 43.56 → 40.56 s，解码 37.37 → 37.87 token/s。两次输出哈希完全一致，均验证 266 轮，热状态 nominal。峰值从 42.75 降至 42.31 GiB，候选系统可用内存最低 50%。短请求、进程互斥、超上下文拒绝和请求后健康检查均通过，优化 HTTP 配置启用按层预填充。见 `runs/verify-panel-prefill-retest.log` 和 `runs/http-panel-prefill.log`。

合批预填充上线后，用同一驻留模型作动态草拟词表 HTTP 前后对照。两轮基线与候选分别为 37.66、38.32、38.00 token/s，端到端为 40.73、40.26、40.55 s。候选从原 BF16 词表补入 20 行，验证 266 → 261 轮。三组 1024 个输出的文本哈希相同，峰值约 42.31 GiB，固定四层 FFN 探针为 3.45–3.62 ms，换页和交换计数不变。首次正式部署样本在 fair 下为 37.08 token/s，超出上线前设定的性能门槛，因此自动回退。该次验证输出、261 轮复现、进程互斥和请求间草拟词表隔离均通过。见 `runs/confirmed-vocab-layer-prefill-http-ab.log` 与 `runs/http-confirmed-vocab-deploy.log`。

把五组原 MTP 单行投影的 BF16 权重按行做可逆编码。内核保留 MLX 的四列顺序累加和 32-lane shuffle 归约，正常和小值应力样本的权重与投影全部逐位一致。编码约占原字节数的 76.7%–76.8%。FFN 输入投影由 0.726 降至 0.660 ms，注意力输入大致持平，另外三组慢于原路径。微基准峰值 6.62 GiB，系统可用内存最低 72%，热状态 fair。见 `runs/mtp-exact-row-gemv-retest.log`。 [MLX 0.32.4 GEMV](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/gemv.h#L150)。

待空闲热状态连续为 nominal 后，再用正式 HTTP 入口复测并启用动态草拟词表。4096/1024 的解码为 39.26 token/s，首 token 为 13.37 s，端到端为 39.43 s。补入 20 行并验证 261 轮，与原 266 轮配置的 1024 个输出文本哈希完全一致。峰值 42.31 GiB，系统可用内存最低 50%。长请求前后的短请求均只加一行，验证次数和输出相同，进程互斥及健康检查通过。见 `runs/http-confirmed-vocab-deploy-retest.log`。

MTP 单行 FFN 输入投影保留框架 SwiGLU 时，同场为 0.741 → 0.674 ms。将 SwiGLU 合进无损行布局内核后为 0.669 ms，但三档较大输入尺度观察到 BF16 元素差异。候选保持为微基准，在线仍使用已验证的 MTP 投影和门控。峰值 6.62 GiB，系统可用内存最低 74%。见 `runs/mtp-exact-row-swiglu.log`。

在已上线动态草拟词表的 4K/1K 配置上，MTP 单行 FFN 只替换权重读取和矩阵向量乘法，继续调用框架原 SwiGLU。真实 MTP 的 10 步复放中隐藏状态、KV 和首选逐位一致。整模按轮交替执行候选投影 1951 次，两个请求的 1024 个输出逐 token 一致，均验证 261 轮。草拟 4.005 → 3.878 s，解码 39.70 → 39.85 token/s，热状态 nominal，计时期间换页和交换计数不变。两路共享整份模型，比较阶段同时持有 340 MiB 原矩阵和 260.86 MiB 编码，解码共同峰值 42.55 GiB。见 `runs/mtp-exact-row-interleaved.log`。

从现行驻留权重中读取主层与输出词表的全部大矩阵缓冲，涵盖 40.09 GB，不含输入嵌入、MTP、KV、状态和小于 1 MiB 的缓冲。纯流式读取并校验中位耗时 56.61 ms，有效吞吐 708 GB/s。相同驻留模型和 4K 缓存的 6/8/11/13 位置完整验证中位数分别为 75.37/78.29/91.68/95.38 ms，各自预测均匹配已记录的后继 token。这个纯读取内核使用独立流式调度，耗时作为实测参考，不能当作物理下界或可直接减去的计算成本。按本轮 261 次验证、约 4.0 s 草拟耗时估算，50 token/s 要求平均主模型验证约 63 ms，已接近纯读取的实测速率区间。整次试验的观测解码峰值 41.54 GiB，可用内存最低 51%，热状态 nominal。见 `runs/full-target-weight-stream.log`。

正式配置在 M2 Ultra 和 MLX 0.32.4 的对应矩阵启用单行 MTP FFN 无损行布局。加载时全量回读原 BF16 位模式校验，只将 340 MiB 原矩阵替换为 260.86 MiB 编码。正常 HTTP 4K/1K 对照解码 39.31 → 39.45 token/s，草拟 3.999 → 3.895 s，端到端 39.50 → 39.31 s。两次 1024 个输出的哈希一致，均为 261 轮。请求峰值 42.31 → 42.24 GiB，候选可用内存最低 51%，热状态 nominal，短请求、互斥与健康检查通过。见 `runs/http-lossless-mtp-ffn.log`。

只用当前提示词的前 3072 个位置拟合 64 维残差方向，尝试校正递归草拟的隐藏状态。剩余 1023 个提示词位置的归一化隐藏状态 MSE 从 1.100 降至 0.836。对同一 1K 响应的 128 个八位置窗口使用已确认 token 做离线复放，平均连续命中从 3.391 降至 3.016，前后两半均下降。校正量仅来自提示词，线上仍使用原草拟递推。本次模型加载及实验峰值 53.93 GiB，系统可用内存最低 52%。见 `runs/mtp-prompt-low-rank-replay.log`。

对四层原版 FFN gate/up 试验向量写回和仅存实际 token 的共享内存。候选保留 FP32 顺序归约及 BF16 SwiGLU 舍入，6、8、11、13、16 位置均逐位一致。6 位置从 0.476 增至 0.491 ms，11 位置从 0.596 降至 0.592 ms，其他位置没有稳定收益。微基准峰值 4.13 GiB，系统可用内存最低 71%，热状态 nominal。线上保持原投影。见 `runs/compact-gate-epilogue.log`。

短批注意力改为相同因果位置的两个查询头共用 K/V。使用与现行内核相同的 128 个分区和 BF16 部分和，6、11、13 位置逐位一致。11 位置、4107 KV 的原路径、按两个 token 共享、按两个头共享分别为 0.636、0.587、0.575 ms；按三个头共享为 0.858 ms。之后让同查询的特征组共用第二阶段 softmax 权重，完整耗时基本持平。将中间 BF16 部分和按每四个特征组分块后，同场 11 位置、4107 KV 为 0.608 → 0.544 ms，13 位置、5101 KV 为 0.772 → 0.669 ms，逐位检查通过。三个微基准峰值不超过 0.08 GiB，可用内存最低 74%，热状态 nominal。见 `runs/gqa-share-query-heads.log`、`runs/gqa-shared-softmax-reduce.log` 和 `runs/gqa-partial-traffic-layout.log`。

按两个查询头共享 K/V、分块存储部分和的候选，在同一份 BF16 模型和 4K 缓存中通过 6、13、11、2 位置的线性、树形和提交状态逐位检查。交替生成 1K 时，候选执行 3712 次，全部输出逐 token 一致，两路均为 261 轮。目标耗时 21.664 → 21.598 s，解码 39.87 → 39.99 token/s。两路峰值 42.22 GiB，可用内存最低 51%，计时期间无新增换页或交换，热状态 nominal。见 `runs/gqa-query-heads-interleaved.log`。

按两个查询头共享 K/V 的正式 HTTP 复测在两轮 nominal 样本中分别跑到 39.46 和 39.38 token/s，端到端分别为 39.41 和 39.39 s。候选未复现整模交替对照的解码提升，服务自动恢复原注意力路径。两次输出哈希一致，均验证 261 轮，峰值均约 42.24 GiB，候选系统可用内存最低 50%。短请求、进程互斥和恢复后健康检查通过。见 `runs/http-gqa-query-heads.log`。

用 4K 提示词前 3072 个位置蒸馏原版目标模型的候选分数，拟合 64 维草拟分数修正，使用每位置目标与草拟各前八名的并集。后 1023 个提示词位置与目标首选的一致数从 582 增至 685，例子采集和拟合合计 2.12 s。对 1K 输出做 128 个连续八位置窗口的离线复放，保持原版 MTP 递推，只改变候选排序。原排序、四分之一修正、半量修正、全量修正的平均连续命中分别为 3.391、3.391、3.375、3.266。修正未启用，采样基线仍复现 1024 个输出 token、同一文本哈希和 261 轮验证。整次峰值 53.93 GiB，系统可用内存最低 51%，原版 BF16 优化配置恢复并通过健康检查。见 `runs/mtp-prompt-rank-distillation.log`。

用上一轮已出结果的主模型验证位置在线校准候选排序，并在下一轮才生效。固定 4K/1K 轨迹得到 1854 组可配对草拟状态与验证标签，其中 878 组来自首个错误候选之后、最终未提交的主链位置。只用已确认前缀学习时修复 9 个首选并损失 18 个，固定轮首估算连续命中由 736 降至 701。并用未提交位置后，修复 12 个并损失 17 个，连续命中为 710。在有空位且原先无备选分支的轮次，后者给出 14 个区别于原第二名的备选，其中 3 个匹配真实续写，同组原第二名命中 2 个。数据为离线顺序回放，修正未上线。生成与回放阶段峰值分别为 41.30 和 40.62 GiB，系统可用内存最低 52%。261 轮基线输出哈希一致，原版 BF16 优化服务已通过恢复后的健康检查。见 `runs/mtp-causal-spine-calibration.log`。

以当前请求的 token ID 为键复用首层 DeltaNet 输入投影。提示词预置 841 行，整表最多 1536 行，占 48 MiB，准备耗时 32 ms。1–16 个位置的 BF16 投影逐位一致，6、11、13、2 位置的线性或分叉整模隐藏状态及提交后的 `KV`、卷积和 DeltaNet 状态也逐位一致。4K/1K 交替对照中 130 轮全部命中，另外 131 轮为 188 个不同 token 补入原版投影。两组输出逐 token 一致，均为 261 轮。累计验证 21.766 → 21.752 s，解码 39.66 → 39.68 token/s，节省时间小于预置成本。保留线上原路径。共用模型的解码峰值 42.31 GiB，可用内存最低 51%，热状态 nominal，计时期间换页和交换计数不变。原版 BF16 服务恢复及健康检查通过。见 `runs/first-delta-projection-cache.log`。

将 9–16 位置 gate/up 的两个候选面板分给不同 SIMD 组，保留原 K 分片和顺序归约。四层实际原版权重的 9、11、13、16 位置之 BF16 SwiGLU 均逐位一致。单独读取面板权重的最快候选在 11 位置由 0.603 增至 0.706 ms。进一步按两包或四包把已还原权重存入线程组共享内存，使两个面板只读取一次设备权重。11、16 位置也逐位一致，11 位置同场原版为 0.598 ms，共享最快为 0.812 ms。两次微基准峰值均为 4.13 GiB，可用内存最低 71%，热状态 nominal。候选留在微基准，原版 BF16 HTTP 服务已恢复并通过健康检查。见 `runs/independent-gate-up-panels.log` 和 `runs/independent-gate-up-shared-packets.log`。

在 DeltaNet 递推 kernel 内，先把短批完整 Q/K 按头归一化，并在同一线程组的值通道间共用。用实际模型形状合成的 6、11、16 位置线性及 13 节点单分叉、双分叉，通过 BF16 输出、FP32 状态、秩一更新与回滚逐位检查。11 位置原路径、只改为每组 16 个 SIMD、同时共享完整 Q/K 分别为 0.229、0.227、0.222 ms。微基准峰值 0.027 GiB，系统可用内存最低 74%。共享的 16 SIMD 方案同时通过整模线性和树回放逐位检查。4K/1K 按轮交替对照中，1024 个输出 token 完全一致，均验证 261 轮。实际目标耗时 21.642 → 22.239 s，解码 39.89 → 39.04 token/s。共同解码峰值 42.22 GiB，可用内存最低 51%，热状态 nominal，计时期间换页和交换计数不变。线上保留原递推。见 `runs/delta-threadgroup-qk.log`、`runs/delta-threadgroup-qk-controls.log` 与 `runs/delta-threadgroup-qk-interleaved.log`。

用同一原版 BF16 HTTP 服务串行运行三个 4096/1024 请求。代码续写为 39.47 token/s、261 轮，中文技术报告为 35.38 token/s、291 轮，平均主模型验证耗时分别为 84.03 与 83.81 ms。中文报告请求用满 128 个已确认词表补充名额，代码请求使用 20 个。英文技术教程为 28.46 token/s、315 轮，该请求的采样热状态由 nominal 变为 fair，平均验证耗时 98.67 ms。三组峰值均约 42.24 GiB，可用内存最低 51%，无新增换页或交换。代码校验文本哈希一致，所有请求前后健康检查通过。见 `runs/http-fixed-length-workloads.log`。

对用满词表补充名额的同一中文输入，把请求级上限从 128 提至 1024。整段输出有 167 个位置、131 种 token 落在初始草拟词表之外。两路采用同一模型和前缀缓存按轮交替，词表补充从 128 增至 131，全部 1024 个 token、291 轮验证及每轮接受量分布相同。热状态全程 fair，此对照解码为 15.70 和 15.69 token/s。共同峰值 42.27 GiB，可用内存最低 51%，无新增换页或交换。线上保留 128 行上限，恢复后健康检查通过。见 `runs/chinese-draft-vocab-capacity.log`。

从中文请求预填充的 252 个位置抽取 BF16 隐藏状态，用原版全词表得分预置最多 384 行草拟词表。准备耗时 0.226 s，采样状态占 2.46 MiB。预置覆盖基线 131 种初次词表外 token 中的 27 种、167 个位置中的 41 个。单份模型交替对照的验证轮数为 291 → 277，解码 35.27 → 36.89 token/s，计入预置耗时后为 29.00 → 27.96 s。共同峰值 42.22 GiB，可用内存最低 51%，计时期间无新增换页或交换，末次采样由 nominal 转为 fair。第 110 个输出 token 有变化，候选保持在试验中。见 `runs/prompt-conditioned-draft-vocab.log`。

将首处分歧前相同的 109 个输出 token 固定为输入，从同一 4K BF16 缓存重放。逐 token 复放时，原调度和预置调度所选 token 的得分分别为 20.000 和 20.125，优先选中预置调度的 token。6 位置分段复放时二者均为 20.125，按 token ID 并列规则选择原调度的 token。该复核只覆盖首处分歧，显示该位置的选词对分批方式敏感。加载和试验峰值 53.93 GiB，可用内存最低 51%，原版 BF16 优化服务恢复并通过健康检查。见 `runs/prompt-vocab-first-difference.log`。

首处分歧在两条实际生成路径上都发生在起点 109、宽度 6 的线性轮首，说明入轮缓存已受此前调度影响。对应最终隐藏向量有 4332/5120 个 BF16 分量不同，平均绝对差 0.05165。原调度两个候选的 BF16 值并列为 20.125，预置调度为 20.125 和 20.250。以原 BF16 词表权重和 FP32 累加结果排序时，两条路径都选 97577。逐 token 和六位置控制复放按 FP32 的胜者也相同。采用未舍入分数的两种草拟调度前 256 token 一致，相对线上 BF16 词表排序于第 40 token 首次变化。见 `runs/prompt-vocab-fp32-difference-retest.log`。

继续对未舍入分数的两种草拟调度做完整中文 4096/1024 交替对照。保持原 BF16 checkpoint 和 FP32 累加顺序，两组 1024 个输出 token 逐个一致，预置词表使验证 260 → 250 轮、解码 39.44 → 41.33 token/s，计入 0.078 s 词表预选后的解码阶段为 25.94 → 24.83 s。共同峰值 42.22 GiB，可用内存最低 51%，全程 nominal，换页及交换计数不变。该数值路径改变了现行 BF16 词表舍入所产生的输出，仍在实验阶段，原版 BF16 优化 HTTP 服务恢复并通过健康检查。见 `runs/prompt-vocab-fp32-interleaved.log`。

在原代码续写 4096/1024 请求上，同样使用原 BF16 权重和未舍入的 FP32 分数排序。预选 190 行，只覆盖对照输出中 14 种初次词表外 token 的 2 种。两组验证分别为 281 和 279 轮，解码 37.02 与 36.94 token/s，计入准备时间约 27.64 与 27.77 s。两组在第 490 个输出 token 首次分歧，并且都与原线上文本不同。峰值 42.21 GiB，可用内存最低 51%，热状态全程 nominal，无新增换页或交换。见 `runs/code-prompt-vocab-fp32-interleaved.log`。

将按逻辑位置归约的祖先注意力覆盖到 2–16 位置，并与未舍入词表排序联用。在共用的代码前缀缓存上，宽度 2、3、5、6、8、13 以及非连续分支相对宽度 11 的共同 token 隐藏状态、预测和提交状态均逐位一致。完整代码 4096/1024 对照中，两种草拟调度生成相同的 1024 个 token，均验证 274 轮，解码 35.24 与 35.31 token/s。两组相对线上哈希都有变化，计入准备开销无净收益，后段热状态从 nominal 转为 fair。共同峰值 42.22 GiB，可用内存最低 51%，换页和交换计数不变。线上保留已验证的原数值路径。见 `runs/code-canonical-attention-fp32.log`。

在逻辑祖先次序内，把同一候选上的两个 GQA 查询头放到同一 SIMD 组，共用 K/V，部分和值按四个特征组排布。2、6、11、13 位置线性、单分叉与非连续祖先组合逐位匹配逐行紧凑参考。11 位置、4465 前缀的同场内核为 0.611 → 0.594 ms，13 位置、5056 前缀的树形内核为 0.723 → 0.691 ms，2 和 6 位置基本持平。含逐行参考的微基准峰值 0.31 GiB，系统可用内存最低 74%，热状态 nominal。见 `runs/gqa-logical-head-pairs.log`。

逻辑 GQA 只在 9 个及以上位置启用查询头配对。整模通过历史和提交状态的逐位检查，执行了 1904 次线性调用和 64 次树调用。与原逻辑内核按轮交替生成代码 4096/1024，同一份 BF16 模型的 1024 个输出 token 逐个一致，均验证 274 轮。目标耗时 22.704 → 22.648 s，解码 38.04 → 38.18 token/s。共同峰值 42.21 GiB，可用内存最低 51%，热状态 nominal，无新增换页或交换。逻辑路径仍保留为试验，默认原版 BF16 HTTP 服务已恢复并通过健康检查。见 `runs/gqa-logical-head-pairs-retest.log`。

用同一份原版 BF16 模型重跑代码、中文报告和英文教程，各 4096/1024，输出哈希和 261/291/315 轮均复现。在仍有同档位空槽且尚无分支的 201/222/238 轮中，事后可找到命中真实续写的第二候选的轮数为 57/56/72。每次整类留出，用另外两类的 MTP 分差、置信度、深度等可见信息拟合分支命中概率。每类选 24 轮时，分类器命中 8/6/4，按原草拟路径累计概率排序命中 5/4/8。每类选 48 轮时分别为 10/8/8 与 10/10/11，学习排序未显示稳定增益。固定轨迹命中只表示可多确认一个候选，不能直接换算为少一次在线验证。最大暂存隐藏状态 17.34 MiB，生成请求峰值不高于 42.26 GiB；原版优化 HTTP 服务恢复并通过健康检查。见 `runs/mtp-branch-risk-probe.log` 和 `runs/mtp-branch-risk-heldout.log`。

将前七个草拟位置的累计 MTP 概率接入实际分支选择。原轮没有分支且第二候选的路径概率至少为 0.25 时，在原 8/16 位置档位内最多加入两个候选。两路共用逻辑位置注意力与未舍入词表排序，代码 4096/1024 的 1024 个输出逐 token 一致。46 条新分支加入 92 个节点，9 条被选中并确认 16 个候选，验证总数仅从 274 降至 273。草拟耗时 4.07 → 5.05 s，目标耗时 22.70 → 22.86 s，解码 38.07 → 36.48 token/s。统一内存峰值 42.23 GiB，系统可用内存最低 51%，热状态 nominal，换页和交换计数不变。该调度保留在实验中，现行 BF16 HTTP 服务恢复并通过健康检查。见 `runs/mtp-same-width-mass-ab.log`。

将每三层连续 DeltaNet 的短线性验证合成一个 MLX 计算图，沿用 6 和 11 位置两个编译宽度及全部 FP32 回滚数据。用同一份 4K 缓存通过 6、13、11、2 位置的隐藏状态、预测及提交状态逐位校验。两种宽度各生成 16 个图，全程执行 2800 次三层图。6、11 位置的主模型建图中位数分别为 1.762 → 1.702 ms 和 1.847 → 1.769 ms，含 GPU 的耗时分别为 71.875 → 71.536 ms 和 87.747 → 87.750 ms。整模 4096/1024 两组输出逐 token 一致，均验证 261 轮，解码为 39.95 与 39.98 token/s。共同峰值 42.22 GiB，可用内存最低 51%，全程 nominal，换页和交换计数不变。收益不足以替换现行图，原版 BF16 HTTP 服务恢复并通过健康检查。见 `runs/compiled-delta-triplets.log`。

FFN down 的 34 个 K 分片先把两个末尾分片分别折入 0/1 号，再复用设备 SIMD 求和的首步相邻合并。对 8192 组合成 FP32 输入，首步配对步长 1 逐位一致，其余候选出现差异。将首步移到投影写回前后，11 位置的完整 FP32 中间缓冲由 7.30 降到 3.44 MiB。四层实际权重的 2、6、11、16 位置完整 FP32 投影逐位一致。同 SIMD 顺序计算两个分片的 11 位置内核由 0.357 增至 0.386 ms。改用两个 SIMD 并行计算、线程组内先合并并分块写回后，11 和 16 位置同场由 0.362/0.365 降至 0.357/0.358 ms，2 和 6 位置较慢。微基准峰值均 1.91 GiB，可用内存最低 72%，热状态 nominal。见 `runs/ffn-down-folded-partials-retest.log` 和 `runs/ffn-down-parallel-simd-fold.log`。

整模只在 9–16 位置启用并行折叠并直接写回 BF16，覆盖 115 轮。6、13、11、2 位置的线性与分叉回放逐位校验通过；代码 4096/1024 的两组输出逐 token 一致，均验证 261 轮。目标耗时 21.733 → 21.900 s，解码 39.76 → 39.52 token/s，未复现独立投影的收益。两组共同峰值 42.22 GiB，可用内存最低 51%，热状态 nominal，换页和交换计数不变。原版 BF16 HTTP 服务已恢复并通过健康检查。见 `runs/ffn-down-parallel-simd-fold-interleaved.log`。

从实际代码 4096/1024 运行中采样五轮，包括早晚段的 6/11 位置线性及 13 节点树。64 层全量回放与各轮线上隐藏状态逐位一致。排除 padding 后，52,363,264 个 BF16 SwiGLU 激活只有 1 个零，174,080 个对应 down 的 32 列包中没有全零包。对每层将幅值最小的一个包置零，只有 8.18%–9.39% 的 BF16 down 输出元素保持逐位一致，没有一个覆盖所有候选的八输出行块保持逐位一致。跳过 1/8、1/4 包的结果更低。该统计仅作可行性诊断，未启用稀疏近似。原 261 轮和文本哈希复现，观测请求峰值 42.24 GiB，可用内存最低 52%，恢复后的 BF16 HTTP 服务通过健康检查。见 `runs/ffn-active-packet-audit.log`。

试验密集三位指数与三个逐 SIMD lane 选择的指数基准。常规载荷按固定索引读取，只有不适用三个基准的 lane 再读原始指数。四层原版 down 权重和 6、11、16 位置的 FP32 部分和逐位一致，另外通过了无回退、全回退和随机 BF16 位模式还原。回退 lane 占比由旧三位布局的 13.1%–14.0% 降至 6.1%–6.2%，载荷约占原 BF16 字节的 74.14%–74.20%，现行四位布局为 76.15%–76.19%。11 位置的原版、旧三位和新密集三位分别为 0.357、0.369、0.408 ms，候选保持在试验中。峰值 3.55 GiB，可用内存最低 71%，热状态 nominal。恢复后的原版 BF16 HTTP 服务通过健康检查。见 `runs/bf16-dense-palette-exponents-retest.log`。

在已记录的 266 轮代码 4096/1024 轨迹上，从请求内此前已见的 token 搜索结构后缀，允许最多五处一致的标识符改名，保留字面标点和源片段不重叠约束。33 次含改名的轮首提案共匹配 45 个候选，原策略在相同轮确认 145 个。固定 8/16 位置档位下，用已知输出事后遍历所有受限候选，含改名的轮首和备选分支最多分别补中 12 个和 9 个候选，涉及 5 轮和 3 轮。保留原 MTP 主链、仅在精确检索与它共享至少四个候选后填入空位，7 次尝试中有 3 轮获得增益，共补中 15 个候选。这是固定轨迹的回放计数，未更改线上策略。CPU 诊断的 Python 分配峰值 35.06 MiB，已部署 BF16 服务仍通过健康检查。见 `runs/code-structural-draft-audit.log`。

用有界整数 token 索引把精确检索接入真实验证，保持 MTP 主链并继续用原 8/16 位置档位。查找只用当前请求已确认且不重叠的源片段，至少匹配原 MTP 主链前四个候选，历史最多覆盖 8192 token。旧轨迹复现 7 次尝试、3 次增益和额外 15 个确认。在当前原版 BF16 4K/1K 按轮交替对照中，实际触发 4 次，其中 2 次加分支，20 个补充候选接受 15 个。验证从 261 增至 263 轮，解码 40.02 → 39.80 token/s，第 766 个输出为首处分歧。共同峰值 42.23 GiB，可用内存最低 51%，计时期间全程 nominal，换页和交换计数不变。线上保留经验证的原策略并通过恢复后的健康检查。见 `runs/mtp-agree-exact-tail-ab.log`。

优化加载器按层需要打开原 safetensors 分片，每个分片只打开一次。每层展开后立即重排并校验权重，释放该层的临时稠密布局。输出词表失去稠密副本后才加载独立的输入嵌入。单模型顺序对照中，MLX 冷启动峰值从 53.93 降到 46.10 GiB，单次加载为 17.03 与 16.76 s，常驻均约 40.47 GiB。正式 HTTP 4096/1024 复测以 261 轮完成，输出哈希与原版本一致，解码 39.35 token/s，首 token 13.41 s。请求峰值 42.24 GiB，系统可用内存最低 52%，前后热状态 nominal，换页和交换计数不变。请求前后的短文本、超上下文拒绝和进程互斥均通过。见 `runs/layerwise-loader-ab.log`、`runs/layerwise-loader-phases.log` 和 `runs/http-layerwise-loader.log`。

从相邻两个已确认位置分别出发，沿同一段真实前缀回放原 MTP，并比较当前状态、早一位置的递推状态以及两种分数混合。代码和中文报告各有 128 个固定八步窗口，均复现 4096/1024 输出哈希及 261/291 轮基线验证。混合后连续命中均值分别从 3.516 → 3.563、2.813 → 2.844，合计只多命中 6 和 4 个位置。两组分别有 10 和 15 个窗口首位未被原前两名覆盖，早一位置的递推首选在这些首位中补中 0 次。该回放沿已确认前缀统计命中，不直接推算在线吞吐。回放阶段峰值 40.72 GiB，系统可用内存最低 63%，全程 nominal，换页和交换计数不变。线上继续使用原草拟策略并通过恢复后的健康检查。见 `runs/mtp-dual-anchor-audit.log`。

把 FFN gate/up 与 down 的 BF16 符号尾数和指数合为一个缓冲，比较按包分平面与按 lane 连续两种布局。两种布局都通过全压缩包、全原样包、四层真实权重的全量还原，以及 6/11/16 位置 FP32 部分和或 BF16 SwiGLU 的逐位核对。四层微基准中最好的按 lane 布局使 6 位置 gate/up 从 0.474 降到 0.451 ms，11 位置 down 从 0.362 降到 0.341 ms。单份模型以逐层替换和还原完成整段 A/B/A，线性和非连续树路径的隐藏状态、预测与提交缓存逐位一致，三次均生成相同的 1024 个 token 并验证 261 轮。解码为 40.08 → 38.76 → 40.01 token/s，累计目标时间为 21.56 → 22.40 → 21.57 s。换布局峰值 43.70 GiB，解码最高 43.44 GiB（含逐位检查保留的缓存快照），可用内存最低 50%，热状态 nominal，换出和交换计数不变。见 `runs/bf16-single-stream-packets.log` 和 `runs/bf16-single-stream-resident.log`。

将 gate/up 和 down 拆开测完整验证，首次 fair 条件下对照本身随时间变慢。空闲热状态稳定后复测，单换 down 的 6、11、13 位置为 74.66、91.63、95.74 ms，前后对照分别为 75.26/75.12、91.42/91.56、96.43/96.13 ms。单换 gate/up 的 11 位置是 93.02 ms，两侧对照是 91.56 和 91.63 ms。分项数据没有显示稳定的整模加速，正式原版 BF16 服务保留原布局并通过恢复后的健康检查。见 `runs/bf16-single-stream-ablation-nominal.log`。

MTP 输入投影按提示词频次缓存最多 128 个 token 在嵌入半边结束时的逐 SIMD lane FP32 累加值，后半边延续原单行 FMA 和归约顺序。命中、未命中的投影和 10 步真实草拟隐藏状态与 KV 均逐位一致。改为每 SIMD 组处理一行后，投影命中由 0.117 降至约 0.068 ms，未命中为 0.170 ms。单份模型交替对照中，代码 4K/1K 命中 1005/1733，中文报告命中 1119/1964，基线和候选分别复现 261 与 291 轮，并逐 token 匹配输出。计入 8.82 与 9.64 ms 建表，代码解码阶段少用 0.10 s，中文少用 0.13 s。峰值 42.35 GiB，可用内存最低 50%，中文末次热状态为 fair，换出和交换计数不变。见 `runs/mtp-exact-lane-prefix-rows1.log`、`runs/mtp-exact-lane-prefix-chinese.log`。 [MLX 0.32.4 GEMV 累加与归约](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/gemv.h#L150)

优化 HTTP 配置启用该缓存后，正式代码 4096/1024 解码为 38.89 → 39.50 token/s，草拟 3.958 → 3.861 s，端到端 39.69 → 39.33 s。两组热状态 nominal，1024 个输出的文本哈希一致，均为 261 轮。两组请求峰值均为 42.24 GiB，系统可用内存最低 51%，换出和交换计数不变。短请求未建表，超上下文拒绝、进程互斥与健康检查均通过。见 `runs/http-mtp-exact-lane-prefix.log`。

用已上线 MTP 累加缓存的正式参数重跑代码 4096/1024，完整复现 261 轮及输出哈希，解码 39.64 token/s。115 个宽轮占用 10.60 s，约为主模型验证的 48.4%。第二个 8 位置面板共含 343 个候选和 577 个填充槽，最终有 25 轮从该面板确认 57 个位置。在保持确认量、轮数和其余阶段耗时相同的假设下，把 115 个宽轮全部按实测窄轮平均 77.39 ms 计费，估算可省 1.70 s，解码约 42.44 token/s。该反事实估算用于衡量宽轮成本，真实删除候选可能损失已确认位置。生成峰值 42.24 GiB，可用内存最低 52%，热状态 nominal，换出及交换计数不变。见 `runs/wide-tail-mma-scalar.log`。

将 FFN down 第一面板留在 SIMD 矩阵乘加，第二面板仅对真实位置做标量 FMA。首次编译暴露 BF16 位模式向量需要用位重解释，修正后用四层真实权重测试 9、11、13 位置，两个面板的 FP32 部分和与最终 BF16 尾部全部逐位一致。11 位置从 0.359 增至 2.798 ms，9 和 13 位置也明显变慢。微基准峰值 1.91 GiB，可用内存最低 72%，热状态 nominal。正式 BF16 服务保留双矩阵面板并在试验结束后通过健康检查。见 `runs/wide-tail-mma-scalar-retest.log`。

在现行 BF16 4K/1K 的代码和中文报告轨迹里逐轮重算原生 MTP 前三名。两组复现原输出哈希与 261/291 轮。240/269 次主链首次拒绝中，正确 token 排第二的有 88/75 次、第三的有 28/15 次，正确与错误首选存在字节前缀关系的分别只有 23/19 次。仅使用草拟时已有的候选和字节内容，在原验证档位仍有空位且无分支的轮次中，前三候选内的同字节替代只补中 7/1 个首位置，未再多确认续写。将原草拟字节用官方 tokenizer 重切分只提出 1/3 条备选，分别确认 0/4 个位置。相同固定轨迹里，按原路径概率挑一个第二候选可补中 24/25 个首位置。该回放只统计候选覆盖，线上保留原策略。两组请求峰值 42.24 GiB，复放峰值 40.53 GiB，热状态 nominal，换出与交换计数不变。最初的双请求采集器在第二次插桩时停止，修正后中文独立复测完成，BF16 服务已恢复并通过健康检查。见 `runs/mtp-byte-boundary-audit.log` 与 `runs/mtp-byte-boundary-chinese.log`。

原生 MTP 多链 GEMV 让同一 SIMD lane 读取一份原版 BF16 权重片段，为每条链保留独立的 FP32 累加寄存器。合批后用各自的 offset、注意力和 KV 缓存，最后合批打分。五组实际投影及 4096/4094/4092 前缀的两链、三链六步递推逐位匹配串行的 BF16 隐藏状态、完整草拟分数、首选与 K/V。两链 20.32 → 12.05 ms，三链 29.64 → 12.94 ms。首次逐投影微基准通过，整链转接因源码缩进匹配停止，修正后完成复测。峰值 6.62 GiB，可用内存最低 71%。见 `runs/mtp-exact-multi-gemv.log`、`runs/mtp-exact-multi-gemv-chains.log`。

保留原根分支阈值和节点安排，将第一步主、备续写合批，沿原有设备同步点读回。单份模型按轮交替的代码 4096/1024 中执行 260 次合批，复用 57 次备选，草拟 3.808 → 3.723 s，解码 39.99 → 40.19 token/s。两路逐 token 一致，均验证 261 轮，峰值 42.29 GiB，可用内存最低 51%，热状态 nominal。扩展试验在同验证档位加入 23 条已合批算出的备选链，命中 6 条、确认 10 个候选，整段验证却从 261 增至 270 轮。第 146 个输出首次分歧，两路实测 39.87 与 38.67 token/s，后段热状态 fair，峰值 42.29 GiB。正式参数保留原根分支规则。见 `runs/mtp-exact-paired-root-interleaved.log` 与 `runs/mtp-exact-paired-root-mass-ab.log`。

精确根分支合批已启用于正式优化 HTTP 配置。首次上线样本在 fair 热状态下为 34.96 token/s；服务空闲恢复 nominal 后，与 nominal 上线前样本相比，代码 4096/1024 解码 39.57 → 39.75 token/s，草拟 3.805 → 3.731 s，端到端 39.18 → 39.25 s。两次输出哈希和 261 轮验证一致，候选 260 次合批并复用 57 个备选续写。两次峰值均为 42.24 GiB，候选可用内存最低 52%，换出和交换计数不变。短请求沿用原路径，超上下文拒绝、进程互斥、恢复后的健康检查均通过。见 `runs/http-mtp-exact-paired-root.log` 与 `runs/http-mtp-exact-paired-root-retest.log`。

按累计 MTP 路径概率跨深度展开待选节点，每批用精确共享权重内核处理一至三条链，最多保留两个新增子分支。代码 4096/1024 原 261 轮的对照与 16 位置候选分别为 40.25 和 37.44 token/s。候选在 247 轮完成，选中 36 条备选路径，接受 57 个备选位置。草拟从 3.723 增至 5.245 s，目标耗时从 21.608 增至 21.937 s，第 7 个输出出现分歧。峰值 42.32 GiB，可用内存最低 51%，热状态 nominal，换出和交换计数不变。见 `runs/mtp-probability-frontier-ab.log`。

将上限收回到 11 个位置，并跳过概率上界低于扩展门槛的节点。整段跳过 203 次叶子打分，草拟为 3.739/3.751 s，目标 21.646/21.155 s，实际解码 40.16 → 40.86 token/s，验证 261 → 249 轮，仍于第 7 个输出分歧。在固定线上输出序列的诊断复跑中，两组都完整计算目标模型，随后按参考 token 选提交位置。原策略无需纠正 argmax，候选在被提交路径上有 9 次 argmax 需要覆盖。这个固定输出对照的验证为 261 → 253 轮，解码阶段耗时 25.57 → 25.25 s。两项复测峰值约 42.27 GiB，可用内存最低 51%，热状态 nominal，换出和交换计数不变。调度仍为实验，原版 BF16 服务恢复并通过健康检查。见 `runs/mtp-probability-frontier-bounded-ab.log` 和 `runs/mtp-probability-frontier-fixed-trace.log`。

追踪 11 节点概率树与现行调度的首处分歧。原全注意力路径在第 4 个共同输入位置、模型第 4 层首次产生激活位差。该 token 在两个 8 节点批中分别是主链节点 0 和侧链节点 5，最大 BF16 激活差为 0.0009765625。从后续两路入轮状态得到的第 7 个输出分别为 262 和 285。完整词表的未舍入 FP32 结果也分别选 262 和 285。将注意力按各节点真实祖先的逻辑位置归约后，原 BF16 排序与 FP32 排序各自的两种调度均匹配最初 48 个输出，采样检查的逐层激活位模式一致。首次跟踪在初始化统计字典时停止，修复后复测峰值 41.83 GiB。见 `runs/mtp-frontier-numeric-retest.log`。

用原版 BF16 权重和 BF16 词表排序，在相同逻辑位置注意力下交替运行现有调度与 11 节点概率树。代码和中文报告分别输入 4096、生成 1024 token，每种负载的两路输出逐 token 一致。代码验证 265 → 252 轮，解码 39.64 → 40.46 token/s；中文报告 281 → 270 轮，解码 37.24 → 37.83 token/s。峰值分别约 42.27 和 42.28 GiB，可用内存最低 51%，换出和交换计数不变，中文末次热状态为 fair。两种负载的逻辑位置结果相对现行 HTTP 输出哈希有变化，本次保留为实验。原版 BF16 服务已恢复并通过健康检查。见 `runs/mtp-frontier-logical-bf16-ab.log` 和 `runs/mtp-frontier-logical-bf16-chinese.log`。

逻辑位置注意力和 11 节点概率树接入同一个驻留模型的 HTTP 请求选项。注意力归约先与原实验内核逐位核对，覆盖三种前缀长度、七种宽度、线性与分叉路径，共 42 组配置。正式 HTTP 的逻辑位置对照中，代码 4096/1024 从 39.23 升至 40.16 token/s，265 → 252 轮。中文报告从 36.82 升至 37.52 token/s，281 → 270 轮。每组输出文本哈希相同，代码默认路径仍复现历史哈希和 261 轮，本次为 39.71 token/s。五个请求峰值均为 42.24 GiB，系统可用内存最低 51%，热状态为 nominal，换出和交换计数保持不变。准入测试模拟低余量，验证额外 220 MiB 计入 Metal 和系统余量，以及清理后仍不足时拒绝。短请求回退、非法调度、超上下文拒绝和进程互斥均通过。见 `runs/logical-attention-integration-micro.log` 和 `runs/http-frontier-mode-retest.log`。

`frontier11` 命令行基准用默认的未包装代码提示生成 1024 token，实际启用概率树，执行 262 轮，解码 23.23 token/s。请求峰值 42.24 GiB，可用内存最低 50%，热状态由 nominal 转为 fair。计划用与 HTTP 相同的输入复测，已逐 token 核对输入。120 秒空闲期间未恢复稳定的 nominal 热状态，本次未开始同输入计时，HTTP 服务保持就绪。见 `runs/frontier-cli-integration.log` 和 `runs/frontier-cli-matched.log`。

在四层真实 BF16 FFN down 权重上测试 K 分片优先的线程组调度，沿用同一份无损权重包和相同的 FP32 累加顺序。首次编译的整数类型冲突修正后，6、8、11、16 位置的 FP32 部分和均逐位一致。包含最终归约的 11 位置微基准从 0.355 降至 0.333 ms，峰值 1.91 GiB，系统可用内存最低 72%。两个整模对照各共用一份 BF16 模型，以 11 节点概率树生成相同的 1024 个 token，均为 252 轮。线性索引版为 40.59 → 40.20 token/s，直接二维网格版为 40.47 → 40.17 token/s。线性及非连续树的隐藏状态、预测和提交缓存逐位一致。两次整模峰值均为 42.25 GiB，系统可用内存最低 51%，热状态 nominal，换出和交换计数不变。服务使用原调度。见 `runs/ffn-down-split-dispatch-retest.log`、`runs/ffn-down-split-dispatch-ab.log` 和 `runs/ffn-down-split-grid-ab.log`。

空闲时连续采到三次 nominal 后，以逐 token 匹配 HTTP 的代码输入重跑 `frontier11` CLI。4096/1024 完成 252 轮，解码 40.07 token/s，首 token 13.36 s，端到端 38.89 s。热状态前后均为 nominal，峰值 42.24 GiB、系统可用内存最低 50%。同输入的既有 HTTP 结果为 40.16 token/s。见 `runs/frontier-cli-matched-nominal-retest.log` 和 `runs/frontier-cli-matched.jsonl`。

在现有 11 节点树上记录代码和中文报告 4096/1024 的每轮首个漏选位置，复现 252/270 轮与历史输出。处于已打分父节点的漏选中，代码和中文分别有 119/178 个真实 token 落在前三名之外，其中 13/133 个当时不在草拟短名单。另有 62/34 个已进入前三名但超过局部分差门槛。用另一份负载的真实前缀回放校准前三名的概率，只从已打分节点补最多三片叶子，保持原窄/宽面板并共享最多两个回滚分叉点。固定轨迹两组各多覆盖 28 个首漏位置。在线交替对照逐 token 一致，代码 252 → 245 轮、40.07 → 40.86 token/s，中文 270 → 269 轮、37.30 → 37.07 token/s。中文后段进入 fair，代码全程 nominal。共同峰值不超过 42.31 GiB，可用内存最低 51%，换出和交换计数不变。见 `runs/frontier-missed-edges.log`、`runs/frontier-shortlist-miss-audit.log`、`runs/frontier-leaf-holdout.log` 和 `runs/frontier-leaf-calibration-ab.log`。

请求内校准只在验证完成后，用已提交路径的完整目标词表首选更新统计。每轮最多补两个已打分的叶子，新叶子无需保存 MTP K/V。两种负载的 1024 个输出均逐 token 一致。代码从 252 降到 247 轮，解码 35.52 → 35.99 token/s。中文从 270 降到 269 轮，解码 23.98 → 23.86 token/s。两份负载全程为 fair，两路共享一份模型，峰值不超过 42.28 GiB、可用内存最低 51%，无新增换出或交换。服务继续使用原 11 节点概率树。见 `runs/frontier-leaf-online-ab.log`。

中文报告复用原验证投影已计算的每个词表块最高分，记录 270 轮的 2441 个目标位置。读取候选用了约 0.20 s。因果回放只用上一轮的分数，每轮最多补四行、八轮未刷新即过期，在固定基线轨迹上覆盖了 133 次词表外首漏中的 75 次。实际交替生成同样的 1024 个输出 token，验证从 270 降到 240 轮，解码从 37.49 升到 41.98 token/s。64 个已确认的新增词表行在确认前已预取，暂存最多 42 行，读取额外分数耗时约 0.15 s。峰值 42.26 GiB，系统可用内存最低 51%，热状态为 nominal，换出及交换计数不变。见 `runs/frontier-target-tile-bank.log`、`runs/frontier-target-tile-replay.log` 和 `runs/frontier-target-tile-vocab-ab.log`。

正式 `frontier_prefetch` HTTP 请求复现中文报告 240 轮和输出哈希。对照解码 37.59 → 41.52 token/s，端到端 40.65 → 38.03 s，两次热状态都是 nominal。代码续写输出哈希一致，验证 252 → 251 轮，观测解码 40.06 → 38.81 token/s，对照期间热状态由 nominal 转为 fair，候选全程 fair。四次请求峰值约 42.24 GiB，系统可用内存最低 51%，换出及交换计数不变。短请求回退、非法调度和超上下文拒绝、进程锁均通过。模拟工作集不足时，词表轮换在读取新权重前拒绝分配。见 `runs/http-frontier-prefetch.log`。

在 240 轮的中文预取路径上逐轮复核真实后继，68 个首漏仍在短名单外，另有 37 个在第 4–8 名。对齐上一轮未提交分支的预测，在固定轨迹上有 19 个首漏可以从该来源找到，按既定节点预算选一个实际覆盖 6 个。对最初 133 个词表外首漏做因果回放时，每轮预取 12 行、保留上限 160 行的候选覆盖 102 个，原来的四行设置覆盖 75 个。随后在单份 BF16 模型上按轮交替生成同样的 1024 个 token，中文 240 → 235 轮、41.53 → 43.18 token/s，确认前已预取的行从 64 增至 87。代码两路均为 251 轮，解码 39.81 → 40.12 token/s。峰值不超过 42.27 GiB，系统可用内存最低 51%，热状态 nominal，换出与交换计数不变。见 `runs/prefetch-true-rank-audit.log`、`runs/prefetch-aligned-hints.log`、`runs/prefetch-intake-replay.log`、`runs/prefetch-wider-intake-ab.log` 和 `runs/prefetch-wider-intake-code-ab.log`。

将较大短名单接入现有 `frontier_prefetch` 后，正式 HTTP 中文报告为 235 轮、42.62 token/s，端到端 37.46 s，热状态前后均为 nominal。代码为 251 轮、38.28 token/s，请求末尾转入 fair。两份输出哈希均复现，词表暂存最高 160/136 行。两次 HTTP 请求峰值 42.24 GiB，系统可用内存最低 51%，换出和交换计数不变。准入仍为预取额外保留 256 MiB，短请求回退、参数拒绝、进程锁和健康检查均通过。见 `runs/http-prefetch-wider.log`。

把分叉状态保存点和恢复槽放进一次计算的路由表，避免按不同分叉位置单独编译 DeltaNet。6、8、11、13 位置的六种拓扑通过 BF16 输出、FP32 递推状态和回滚的逐位校验。用原版 BF16 权重对四种 8/11 位置整模拓扑检查了隐藏状态、预测和提交缓存。中文 4096/1024 中有 127 轮使用新编译路径，两路都以 235 轮生成逐 token 相同的输出。预热对照为 43.16 → 43.58 token/s，计入首次建图的复测为 42.67 → 43.45 token/s。两次解码峰值均约 42.26 GiB，系统可用内存最低 51%，无新增换出或交换。新路径在当前模型生成 192 份图，按两个宽度和两种分叉数分组。见 `runs/tree-delta-prepared-micro.log`、`runs/tree-delta-compiled-ab.log`、`runs/tree-delta-compiled-cold-ab.log` 和 `runs/tree-delta-compile-integration.log`。

编译分叉接入 `frontier_prefetch` 后，通过正式 HTTP 复现中文 235 轮、代码预取 251 轮和默认路径 261 轮的历史输出哈希。三次解码分别为 42.44、39.98 和 39.56 token/s。中文 HTTP 单次结果略低于此前的 42.62 token/s，提速幅度以同场交替对照为准。三次请求峰值均约 42.24 GiB，系统可用内存最低 51%，热状态为 nominal，换出和交换计数不变。短请求回退、参数拒绝、进程锁和健康检查通过。见 `runs/http-compiled-tree.log`。

测试两处词表归约。目标词表的块内归约把 11 位置读回从 0.566 降至 0.482 ms，完整词表首选与并列规则通过核对。首次 Metal 编译的保留字冲突修正后通过对照。草拟归约用 SIMD 局部前三名替换三次完整短名单搜索，保留原 BF16 排名、并列时的行顺序和 FP32 `logsumexp`。1–3 行和 32,768–33,617 个词表位置的合成对照逐位一致，排序耗时约缩短 0.04 ms。预热后三路共用一份 BF16 权重完成中文报告 4096 输入 / 1024 输出的负载，235 轮及输出逐 token 一致。原流程、目标词表归约和草拟归约分别为 43.05、43.25、43.30 token/s，草拟从 3.903 降至 3.789 s。三份请求缓存共存时峰值 42.77 GiB，可用内存最低 51%，热状态由 nominal 转为 fair，无新增换出或交换。草拟归约已接入概率树，目标词表归约保留在实验中。见 `runs/vocab-tile-reduce-micro-retest.log`、`runs/frontier-rank-reduce-integration.log` 和 `runs/verification-reductions-warm.log`。

正式草拟归约启用后，空闲时确认稳定 nominal，再通过 HTTP 复测两份 4096/1024 输入。中文为 43.02 token/s、235 轮，端到端 37.26 s。代码为 40.17 token/s、251 轮，端到端 38.91 s，请求末尾转为 fair。历史输出哈希和预取确认行数均复现，两个请求峰值约 42.24 GiB，可用内存最低 51%，无新增换出或交换。短请求回退、超上下文拒绝、进程锁及健康检查通过。见 `runs/http-frontier-rank-reduce.log`。

在每个草拟树前缀上查找 3、4、6、8 token 的历史后缀。每个键最多保留 12 次出现的位置。续写须在已观察的样本中取得一致，三 token 后缀要求至少两次支持。追加候选不占 MTP 状态，也不需要新开第二面板。固定轨迹中最多三个候选可多覆盖中文 55 个、代码 41 个 token，延伸至五个时为 80 和 63 个。通过删除 MTP 叶子保持原宽度时，两种负载的净确认量都下降。只为历史候选开第二面板的回放同样命中较少。5120 个无重复 token 的索引测试峰值 3.57 MiB。见 `runs/frontier-phrase-replay.log`、`runs/frontier-phrase-budget-replay.log`、`runs/frontier-phrase-panel-replay.log` 和 `runs/frontier-phrase-length-replay.log`。

单份 BF16 模型交替验证 4096/1024，三 token 短语把中文从 235 降至 224 轮、43.05 提至 44.84 token/s，把代码从 251 降至 246 轮、40.13 提至 40.75 token/s。三和五 token 的再次对照中，中文从 224 降至 220 轮，44.66 提至 45.62 token/s。代码保持 246 轮，38.26 至 38.41 token/s，两路均处于 fair。所有输出逐 token 一致，统一内存峰值最高 42.34 GiB、可用内存最低 51%，无新增换出或交换。接入前在 3420 次历史查询上核对了正式实现和五 token 原型。见 `runs/frontier-phrase-ab.log` 和 `runs/frontier-phrase-extended-ab.log`。

重启本地 HTTP 服务后，`optimized` 的 `frontier_prefetch` 以原版 BF16 完成两次 4096/1024 请求。中文 45.44 token/s，220 轮，端到端 35.95 秒。代码 40.36 token/s，246 轮，端到端 38.81 秒。两个输出 SHA-256 均与接入前一致。两次测得 MLX 峰值约 42.24 GiB，系统可用内存最低 51%，测试前后 pageout、swapin、swapout 计数一致。请求结束后 `/healthz` 和短请求检查通过。见 `runs/http-frontier-phrase.log`。

以四个原生 MTP 首选作为长短语的使用条件。单份 BF16 模型交替运行 4096/1024 时，中文解码从 45.40 升到 46.76 token/s，验证从 220 降至 217 轮，触发 11 次。代码从 40.28 升到 40.71 token/s，验证从 246 增至 247 轮，触发 9 次，结束时热状态为 fair。两组输出逐 token 一致，验证宽度仍为最多 16 个位置。共同峰值 42.35 GiB，系统可用内存最低 51%，无新增换出或交换。从四个 MTP 首选之后另查历史续写的候选，在固定旧轨迹上少确认 3 至 5 个 token。用空位加第二条短语时最多多验证 26 个位置，额外确认 0 个 token。两项均停在 CPU 回放。见 `runs/frontier-phrase-root-anchor.log`、`runs/frontier-root-phrase-ab.log`、`runs/frontier-suffix-anchor-replay.log` 和 `runs/frontier-two-phrases-replay.log`。

将 16 位置纳入紧凑分叉的编译范围。四种拓扑的隐藏状态、预测和提交缓存逐位一致。预热后单份 BF16 模型交替生成 4096/1024，中文解码从 46.63 升至 46.77 token/s，代码从 39.89 升至 40.44 token/s，验证轮数和输出均一致。热状态为 fair，共同峰值最高 42.35 GiB，系统可用内存最低 51%，无新增换出或交换。见 `runs/frontier-wide-delta-ab.log`。

服务重启后首次运行预取模式，中文 4096/1024 解码 46.18 token/s，217 轮，首 token 13.46 秒，端到端 35.61 秒，热状态从 fair 转为 nominal。代码解码 40.39 token/s，247 轮，首 token 13.45 秒，端到端 38.77 秒，热状态从 nominal 转为 fair。历史输出哈希和长短语触发次数均复现。请求峰值约 42.24 GiB，系统可用内存最低 51%，无新增换出或交换。短请求回退、非法调度、超上下文拒绝、进程锁和请求后的健康检查均通过。见 `runs/http-frontier-anchored.log`。

逐轮审计上述两种负载。在首个漏选位置，真实 token 已在 MTP 第二或第三名的，中文有 48 次，代码有 83 次。仍在草拟词表外的分别有 39 和 4 次。固定轨迹回放还将原有长短语尝试延伸至 24 或 32 个位置，中文额外确认 0 个 token，代码分别净多确认 32 和 33 个 token，完全覆盖后续 2 和 3 轮。回放只核对候选覆盖，暂不扩宽现行验证。见 `runs/frontier-current-audit-retest.log` 和 `runs/frontier-wide-anchor-replay.log`。

在同一份 BF16 模型上交替生成，按请求内已提交路径的命中情况校准分支优先级。中文解码 46.17 → 47.10 token/s，验证 217 → 215 轮，热状态 nominal。代码解码 38.37 → 39.35 token/s，验证 247 → 242 轮，热状态 fair。两组始终逐 token 一致，共同峰值最高 42.35 GiB，系统可用内存最低 51%，无新增换出或交换。见 `runs/frontier-path-calibration-ab.log`。

正式 HTTP 启用路径校准后，中文和代码 4096/1024 的解码分别为 46.70 和 40.96 token/s，端到端分别为 35.32 和 38.38 秒。两个请求分别收集 867 和 913 个已提交路径样本，复现 215 和 242 轮以及原有输出哈希。两个请求的热状态均为 nominal，峰值约 42.24 GiB，系统可用内存最低 51%，换出和交换计数保持不变。短请求回退、非法调度、超上下文拒绝、进程锁和健康检查均通过。见 `runs/http-frontier-calibrated.log`。

用原版 BF16 草拟分数保留第四名，将校准后的路径概率同时用于筛掉低分候选。两种 4096/1024 负载均复现原输出。中文在单份模型交替对照中从 46.64 升到 47.88 token/s，验证从 215 降到 211 轮。代码从 40.87 升到 42.17 token/s，验证从 242 降到 238 轮。两组新增的低原生分数候选为 76 和 81 个，其中第四名候选在整段各选中 3 个。共同峰值最高 42.35 GiB，系统可用内存最低 51%，均处于 nominal，无新增换出或交换。见 `runs/frontier-calibrated-reserve-ab.log`。

正式 HTTP 的排名和 BF16 并列规则先通过逐位核对。首次重启后的中文样本为 37.54 token/s，目标验证和草拟都出现计时波动。使用同一驻留服务预热后，中文为 47.53 token/s、211 轮、端到端 34.98 秒，代码为 41.79 token/s、238 轮、端到端 37.89 秒。两个请求均复现历史输出哈希，MLX 峰值 42.24 GiB，系统可用内存最低 51%，换出与交换计数一致。服务健康和请求准入检查通过。见 `runs/http-frontier-reserve.log` 和 `runs/http-frontier-reserve-warm.log`。

按历史验证耗时给宽批设置收益门槛。仅根据达到八位置时的待选队列，估计额外三个原生位置的确认数。单模型交替对照中，中文在 83 次可扩宽轮里保留 15 次窄批，解码 46.55 → 47.47 token/s，验证 211 → 212 轮。代码在 114 次可扩宽轮里保留 32 次窄批，解码 34.20 → 35.17 token/s，验证 238 → 242 轮。两组输出逐 token 一致。中文热状态转为 fair，代码全程 fair。共同峰值不超过 42.36 GiB，系统可用内存最低 51%，无新增换出或交换。见 `runs/frontier-width-value-replay.log`、`runs/frontier-width-forecast-replay.log` 和 `runs/frontier-width-forecast-ab.log`。

在逻辑位置注意力的归约中直接做 BF16 门控并写出矩阵片段，六种线性和分叉形状逐位匹配原片段。8、11、16 位置耗时分别由 0.48996、0.64440、0.76706 ms 增至 0.49527、0.65664、0.78117 ms。微基准峰值 0.076 GiB，验证器继续使用原路径。见 `runs/fused-logical-gate-pack-micro.log`。

宽度决策接入正式 HTTP 后，中文和代码 4096/1024 的解码为 47.68 和 42.03 token/s，端到端为 34.88 和 37.75 秒。复现 212 和 242 轮、15 和 32 次保留窄批以及既有输出哈希。两个请求峰值约 42.24 GiB，系统可用内存最低 51%，无新增换出或交换。中文全程为 nominal，代码末尾转为 fair。短请求回退、非法调度、超上下文拒绝、进程锁和请求后的健康检查均通过。见 `runs/http-frontier-width.log`。

逐轮复现 212 和 242 轮的正式 4K/1K 轨迹。沿已有草拟路径拼接历史短语、改按路径概率选择短语，以及从内部节点尝试长短语，均未在固定轨迹上增加净确认量。对既有主模型验证结果做因果回放时，按四分之一权重加入未提交分支的首选预测，使后续已提交节点的总负对数似然由 549.80 降至 532.90，由 702.46 降至 657.18。见 `runs/frontier-live-phrase-trace.log`、`runs/frontier-history-overlap-replay.log`、`runs/frontier-inner-anchor-replay.log` 和 `runs/frontier-counterfactual-calibration-replay.log`。

在同一份 BF16 模型上按轮交替，低权重纳入未提交分支后，中文解码由 47.70 升至 48.55 token/s，代码由 41.94 升至 42.83 token/s。验证分别为 212 → 213 和 242 → 244 轮，保留窄批分别由 15 增至 19 次、32 增至 49 次。输出逐 token 一致，最高峰值 42.36 GiB，系统可用内存最低 51%，热状态 nominal，无新增换出或交换。见 `runs/frontier-counterfactual-calibration-ab.log`。

只对较强的历史续写将 MTP 比对提前至两 token。单模型交替的中文和代码各提前收尾六次，解码分别为 47.76 → 48.50 和 42.16 → 42.75 token/s，验证分别为 213 → 213 和 244 → 243 轮。两组输出逐 token 对齐，峰值最高约 42.35 GiB，系统可用内存最低 51%，无新增换出或交换。代码末尾热状态转为 fair。见 `runs/frontier-early-root-replay.log` 和 `runs/frontier-early-root-ab.log`。

两项改动接入正式 HTTP 后，中文和代码 4096/1024 解码分别为 47.80 和 42.55 token/s，端到端分别为 34.84 和 37.47 秒，复现 213 和 243 轮及两个历史输出哈希。两组分别纳入 704 和 888 个低权重样本，并各提前结束 MTP 展开六轮。两个请求峰值约 42.24 GiB，系统可用内存最低 51%，无新增换出或交换。中文全程 nominal，代码末尾转为 fair。参数拒绝、短请求回退、进程锁及请求后的健康检查通过。见 `runs/http-frontier-counterfactual.log`。

当验证批达到八位置时，用未评分父节点的概率上界预判扩宽是否还有收益。三路交替对照中，提前在中文 18 轮、代码 59 轮结束草拟，分别少评分 29 和 81 个位置。解码由 47.79/43.08 提到 48.62/43.13 token/s，验证仍为 213/243 轮。将未提交分支标签权重提高到一半时，中文/代码为 47.97/42.93 token/s，正式采用原四分之一权重。三路共享一份模型，最高峰值约 42.88 GiB，系统可用内存最低 51%，无新增换出或交换。正式 HTTP 复现 18/59 轮提前结束、29/81 个评分节省及原输出哈希，解码为 47.99/42.81 token/s，峰值 42.24 GiB。见 `runs/frontier-widen-bound-ab.log` 和 `runs/http-frontier-widen-bound.log`。

短批逻辑祖先注意力对已经对齐的 BF16 查询、键和值使用 16 字节向量读取，保留原 FP32 展开和归约顺序。五组跨度 4096–5081 的线性或分叉形状逐位匹配，6 位置单层由 0.439 降至 0.381 ms，16 位置由 0.782 降至 0.666 ms。整模交替对照中，中文/代码解码由 47.85/42.67 提到 49.34/44.03 token/s，验证仍为 213/243 轮，1024 个输出均逐 token 一致。正式 HTTP 为 48.43/43.54 token/s，端到端为 34.51/36.92 秒，原输出哈希一致。峰值约 42.24 GiB，系统可用内存最低 51%，换出与交换计数未增加，服务通过健康检查。见 `runs/gqa-vector-kv-probe.log`、`runs/gqa-vector-kv-ab.log`、`runs/vector_kv_production.log` 和 `runs/http-frontier-vector.log`。

将 DeltaNet 的四抽头卷积、BF16 SiLU 和 Q/K 的 FP32 归一化放在同一个 kernel，按原 SIMD 累加顺序归约。合成实际形状的五种线性和分叉输入中，BF16 输出、FP32 状态、秩一更新和归一化 K 均逐位一致。卷积加递推在 8 和 16 位置树形上分别由 0.270/0.302 ms 降至 0.252/0.275 ms，峰值小于 0.015 GiB。见 `runs/delta_conv_qk_norm.log`。

DeltaNet 融合在同一份原版权重的整模回放中通过五种线性和分叉形状的隐藏状态、主模型首选及提交缓存逐位核对。4K/1K 交替对照的中文为 49.07 → 48.63 token/s，代码为 43.91 → 43.91 token/s。生成结果和验证轮数相同，峰值约 42.74 GiB，系统可用内存最低 51%，无新增换出或交换。正式服务使用原 DeltaNet 路径。见 `runs/delta-conv-qk-norm-ab.log`。

在已启用 Q/K/V 向量载入的版本上，另将 BF16 注意力中间结果改为 16 字节向量写回和读入。五组形状通过逐位整数比较。两组 8 位置基本持平，11/16 位置各只缩短约 0.006/0.008 ms，6 位置增加约 0.013 ms。微基准峰值约 0.068 GiB，测试后服务健康检查通过。见 `runs/gqa-vector-partial-probe.log`。

在先前验证 213/243 轮的两种 4K/1K 轨迹上，八位置主模型验证中位数约 75.0 ms，十六位置约 94.1 ms。两组各有 62 次通过队列决策扩宽，其中 27/39 次未确认任何扩宽后新增的位置。假设能提前识别这些轮次且确认序列相同，仅计算目标验证成本，估算可减少 0.50/0.71 秒。见 `runs/frontier-width-audit-current.log` 和 `runs/frontier-wide-oracle-current.log`。

继续用一份 BF16 模型交替测试分叉状态复用和扩宽门槛。放宽到两个不同分叉父节点时，中文/代码为 49.49/43.45 token/s，对照为 48.57/43.95。按宽批实测成本固定 0.95 门槛时为 48.50/44.57 token/s。进一步限制只复用已经保存的分叉状态时，中文/代码为 49.66/43.37，对照为 48.48/43.93。按请求已确认收益动态调整门槛时，两组为 49.42/43.96，中文批次和评分计数与对照一致。各对照输出逐 token 一致，分叉和扩宽策略继续使用生产基线。三路的模型峰值不超过 42.89 GiB，系统可用内存最低 51%，无新增换出或交换。见 `runs/frontier-fork-budget-ab.log` 和 `runs/frontier-adaptive-width-ab.log`。

在已驻留模型上，用合成状态运行 48 层 DeltaNet 的八个线性和紧凑分叉图。首遍合计 0.865 秒，第二遍 0.573 秒，首次图准备差额约 0.292 秒。探针峰值约 40.50 GiB，系统可用内存最低 58%，换出与交换计数不变。优化服务预留 1 GiB 余量后在就绪前预热这些图，合成状态在就绪前释放。见 `runs/prewarm-delta-verify-probe.log`。

正式优化服务的预热耗时 0.864 秒，进程启动至健康检查就绪为 18.38 秒。其后中文/代码 4096/1024 解码为 48.95/43.54 token/s，端到端分别为 34.36/36.94 秒。验证仍为 213/243 轮，两个原输出哈希一致。请求峰值约 42.24 GiB，系统可用内存最低 51%，热状态均 nominal，无新增换出或交换，服务健康检查通过。测试也确认内存余量不足时在合成缓冲区分配前跳过预热。见 `runs/http-frontier-prewarm.log`。

将单行 MTP 的 FFN、尾部归一化与固定 32K 草拟词表投影编入同一张图，逐位核对前六步的隐藏状态、词表分数与 K/V。中文和代码交替生成的输出均逐 token 一致，草拟时间分别为 3.035 → 3.033 秒和 3.435 → 3.432 秒。整段计时的差异主要发生在使用同一实现的目标验证中。候选保留为实验，单份模型峰值不超过 42.36 GiB，可用内存最低 51%，服务恢复后通过健康检查。见 `runs/mtp-compiled-tail-ab.log`。

在当前轨迹上用一至两个 token 的后缀和重复出现的续写补已有批次空位，并保持分叉上限。最宽松的两 token 规则在中文与代码只多确认 1 和 6 个位置，覆盖不足以进入整模试验。回放只使用 CPU。见 `runs/frontier-short-context-replay.log`。

提前读取下一组逻辑位置的 BF16 K/V，保持原有 FP32 运算顺序。五种线性与分叉形状均逐位一致，但 6/11/16 位置分别由 0.388/0.523/0.683 增至 0.432/0.585/0.793 ms。微基准未加载模型，峰值约 0.068 GiB，可用内存最低 74%，热状态 nominal。正式服务恢复后通过健康检查。见 `runs/gqa-kv-pipeline-probe.log`。

对两组当前 4K/1K 负载记录 MTP 前八名和各已验证节点的目标首选，调度仍只使用前四名。复测逐轮复现原候选、分支、得分和输出，验证为 213/243 轮。首个漏选中，正确 token 排第五至第八名的各有 25 次；还有 64/65 次落在八名之外，60/42 次未对该父节点打分。已知正确答案时，24/21 次可通过交换叶子补入原档位；按先前轮次校准分数执行的回放中，两组净确认增量均为零。模型峰值 42.24 GiB，系统可用内存最低 51%，无新增换出或交换。首次诊断在创建插桩环境时停止，修正后完整复测并恢复服务。见 `runs/frontier-rank8-diagnosis-retest.log` 和 `runs/frontier-rank8-replay.log`。

在固定轨迹上，按历史旁支的目标首选建立有界后缀索引，在允许的验证档位内替换候选，中文和代码净多确认 1 和 -1 个 token。另测预留一个草拟步骤扩展未打分的叶子，分别提出 20 和 23 次，只有 2 和 4 次抵达实际确认的末端，尚未计入新增草拟成本。两个回放仅使用 CPU。见 `runs/frontier-shadow-suffix-replay.log` 和 `runs/frontier-terminal-lookahead-replay.log`。

使用最新正式 HTTP 的总解码时间、累计目标时间和验证轮数估算，50 token/s 还需为中文和代码各省约 0.44 和 3.04 秒。若单轮目标成本及草拟等其余开销固定，达到 50 token/s 对应最多 207/205 轮，达到 100 对应最多 83/80 轮。此为预算估算，实际开销会随批宽、验证轮数和接受率变化。实测数据见 `runs/http-frontier-prewarm.log`。

将草拟的固定前缀从 32K 改成 16K，保留提示词与请求内词表预取。在固定轨迹的已提交路径上，16K 加提示词和此前目标预测会漏掉 51/9 个原先在草拟前四名中的正确 token，其中 29/4 个原先为首选。整模对照额外挂载约 122.39 MiB 无损 BF16 前缀，权重和 1–3 行投影逐位相同。首轮 nominal 对照中，中文为 48.85 → 49.53 token/s，代码为 44.11 → 43.78 token/s，两组草拟各省约 0.17/0.19 秒。双路预热后，中文为 48.04 → 48.32 token/s，草拟从 3.084 降至 2.906 秒。代码进入 fair，解码为 12.94 → 13.11 token/s，不能用此次热态复测推断 nominal 下的吞吐。两次对照均逐 token 匹配原输出，中文验证为 213 → 216 轮，代码均为 243 轮。解码对照峰值不超过 42.48 GiB，可用内存最低 51%，无新增换出或交换。正式服务继续使用原 32K 草拟配置，恢复后健康检查通过。见 `runs/frontier-context-vocab-replay.log`、`runs/frontier-compact-context-vocab-ab.log` 和 `runs/frontier-compact-context-vocab-warm-ab.log`。

在四层真实 gate/up 权重上比较 K 分片优先调度与原线程组内归约。新路径在独立线程组生成 FP32 部分和，再按原顺序合并。加上同一线程组四输出块交错排布后，6/8/11/16 位置的单层耗时由 0.475/0.478/0.592/0.598 降至 0.466/0.473/0.583/0.586 ms，权重回读和输出逐位一致，峰值 4.89 GiB。整模对照先隔离调度并共用原权重布局，在同一代码 4K 缓存上验证了四种形状的隐藏状态、预测和提交缓存。6 位置线性与 8 位置树从 73.54/74.95 增至 75.23/76.65 ms，11 位置线性和 16 位置树基本持平。整模对照峰值 41.57 GiB，可用内存最低 50%，无新增换出或交换，服务恢复后通过健康检查。见 `runs/gate-up-split-stream-probe.log`、`runs/gate-up-split-layout-probe.log` 和 `runs/gate-up-split-verification-ab.log`。

从五层原始 checkpoint 的 gate、up、down 读取有界行样本，指数的经验熵约为 2.56 位。相邻指数对采用每对 5/6 位的字典并按 lane 保留原始异常指数时，估算大小分别约占原权重的 84.0%/74.4%。这是载荷估算，未测试 Metal 解码速度。此外，现行 73/69 个宽批中，已知真实路径时 36/49 个可缩到八位置。按两档验证耗时差、假定层耗时均匀估算，在半层处裁剪的理想节省为 0.34/0.46 秒，未扣除同步、判断和缓存重排开销。两项分析均只使用 CPU。见 `runs/bf16-exponent-pair-audit.log` 和 `runs/frontier-staged-verification-oracle.log`。

在 M2 Ultra 的 Metal TensorOps 回退路径上，16×32×16 的原生 BF16 张量输入与协作张量输入均通过等值检查。修正模板调用和输入类型后，8×8×8 仍受 TensorOps 分块约束拒绝。用两层真实 down 权重，将每个 512 列分片改用 TensorOps，8/16 位置的 FP32 部分和及最终 BF16 投影都与现行无损内核逐位一致。将原始 BF16 权重改为连续 32 输出列的小块后，8 位置从 0.519 降至 0.451 ms，16 位置从 0.612 降至 0.555 ms，同场现行内核分别为 0.319 和 0.424 ms。微基准不加载整模，峰值 1.99 GiB，可用内存最低 72%，没有新增换出或交换。见 `runs/mpp-bf16-capability-retest.log`、`runs/mpp-bf16-down-probe.log`、`runs/mpp-bf16-down-layout-probe.log`。 [Apple TensorOps 对较早 GPU 的支持](https://developer.apple.com/videos/play/tech-talks/111432/)

在两层真实权重上将无损 BF16 权重包还原为 32×32 共享内存操作数，直接交给 TensorOps。8/16 位置的 FP32 分片和与最终 BF16 写回均逐位一致，耗时从 0.333/0.419 增至 0.793/0.972 ms。直接在协作张量寄存器解包的另一路触发 GPU 命令缓冲失败后退出，服务恢复并通过健康检查。共享内存路径峰值 1.83 GiB，可用内存最低 72%，无新增换出或交换。见 `runs/mpp-lossless-down-retest.log`。

固定轨迹回放分别给经过非首选祖先的 MTP 节点、重复的草拟至目标词序列建立因果统计。祖先条件在已提交路径上的负对数似然，中文由 567.97 变为 572.76，代码由 722.69 变为 719.42。按当前 token 与 MTP 首选联合统计后，中文/代码为 566.44/719.76，代码的首选准确数由 664 降至 661。用窄批预计确认数和两档实测成本决定扩宽，对原轨迹估算只省约 0.05/0.08 秒。这些结果来自 CPU 回放，服务使用现行调度参数。见 `runs/frontier-ancestral-rank-replay.log`、`runs/frontier-token-confusion-replay.log` 和 `runs/frontier-throughput-width-replay.log`。

本机优化服务使用 `--draft-schedule frontier_prefetch` 启动。省略调度字段的中文和代码 4096/1024 请求分别为 48.97 和 43.71 token/s，HTTP 响应耗时 34.33 和 36.84 秒，输出哈希与显式快路径的既有结果一致，分别验证 213 和 243 轮。代码显式指定 `standard` 时复现该路径原哈希和 261 轮，本次为 39.67 token/s、39.24 秒。三组为 nominal，MLX 峰值约 42.24 GiB，可用内存最低 51%，无新增换出或交换。省略调度的短请求回退、非法调度、超上下文拒绝、进程互斥和请求后健康检查均通过。见 `runs/http-default-frontier-prefetch.log`。

同一驻留服务另测英文技术教程 4096/1024。省略调度字段时实际启用 `frontier_prefetch`，完成 277 轮，解码为 39.56 token/s。热状态 nominal，MLX 峰值约 42.24 GiB，可用内存最低 52%，无新增换出或交换。请求前后的健康检查均显示快路径已配置为本机服务默认。见 `runs/http-default-frontier-english.log`。

短批 GQA 分别试验用一个指数替代在线 softmax 的两个指数、让同一 SIMD 组依次计算两个原分区以复用查询，以及在寄存器中共用两个特征组的归约权重。五种形状和五档缩放均逐位一致。单指数版在 6–16 位置与原路径持平或略慢，后两种在 8–16 位置也更慢。合成缓冲峰值不超过 0.131 GiB，可用内存最低 74%，服务恢复后通过健康检查。见 `runs/gqa-single-exp-probe.log` 和 `runs/gqa-partition-feature-probe.log`。

同一 BF16 模型重新采集英文、中文、代码的当前验证轨迹和目标词表块间分差，复现 277/213/243 轮及三个正式输出哈希。按上一轮分差给下一轮草拟校准分组，三组的已提交路径负对数似然均增大。按旁支分差调整旁支票重只有小幅变化，三个首漏位置子集的首选命中数均保持一致。英文/中文/代码各有 67/64/65 次已打分的首漏落在前八名之外，其中 52/1/55 次的正确 token 已在固定 32K 前缀。继续把已打分的叶子填入同档位空位，分别多确认 1/0/1 个位置。轨迹采集峰值约 42.24 GiB，可用内存最低 51%，无新增换出或交换。见 `runs/frontier-target-margin-audit.log`、`runs/frontier-target-margin-replay.log`、`runs/frontier-three-workload-coverage.log` 和 `runs/frontier-paid-leaf-replay.log`。

直接在物理词表顺序归约 BF16 分数，同时以逻辑词表 ID 处理并列。四种批宽的随机分数、正负零、无穷值、跨分区并列和抽样 NaN 输入与现行 argmax 一致。11 行微基准原路径、两段归约和单线程组归约分别为 0.325、0.208 和 0.239 ms。同一份模型、同一前缀缓存交替验证时，两段方案的中文/代码/英文为 49.69/44.09/32.80 token/s，对照为 49.44/44.15/32.77。三组逐 token 一致，验证轮数仍为 213/243/277，英文末段为 fair。整模峰值最高 42.36 GiB，可用内存最低 51%，无新增换出或交换。正式服务保留现行归约，已恢复并通过健康检查。见 `runs/vocab-exact-tile-argmax-probe.log`、`runs/vocab-exact-tile-single-probe.log` 和 `runs/vocab-exact-tile-interleaved.log`。

在原验证宽度内，依照已有 MTP 分数重分配草拟树。根据真实输出事后挑选前四名中的叶子，英文/中文/代码固定轨迹最多多确认 108/51/82 个位置。按现行概率做树内最优分配时，实际净增 3/0/2 个位置。用另外两组数据拟合每条路径的到达概率，再对留出组分配，净增 7/1/0。整模把留出拟合用于建树时，英文和代码验证从 277/243 减到 273/241，中文从 213 增到 216，输出逐 token 一致。近邻草拟分数投票未补中前三组原先漏掉的首选。见 `runs/frontier-tree-repack-replay.log`、`runs/frontier-path-probability-holdout.log`、`runs/frontier-path-probability-interleaved.log` 和 `runs/frontier-neighbor-logits-replay.log`。

采用三组留出权重的平均值，并根据请求内已验证末端的第二至第四名命中率决定是否启用排序修正。单模型按轮交替对照中，代码由 44.16 升至 44.48 token/s，英文由 37.75 升至 38.59，中文由 33.51 升至 33.61；验证由 243/277/213 变为 241/271/213。三组输出逐 token 一致，分别启用 205/213/10 轮。代码中途转入 fair，后两组开始时已处于 fair。峰值最高 42.36 GiB，可用内存最低 51%，无新增换出或交换。见 `runs/frontier-adaptive-reach-interleaved.log`。

默认 HTTP 入口复测 4096 输入 / 1024 输出。中文技术报告、代码、英文技术教程分别达到 49.02、43.99、40.59 token/s，首 token 延迟为 13.44、13.47、13.52 s。三组输出 token 哈希与原 BF16 路径一致，分别验证 213、241、271 轮。中文与英文开始和结束均为 nominal，代码结束时为 fair。峰值约 42.24 GiB，系统可用内存最低 51%，pageout 和 swap 计数未增长。按当前实测，三组分别还需将解码缩短 0.41、2.80、4.75 s 才达到 50 token/s。见 `runs/http-adaptive-reach.log`。

对概率至少 0.65 的已确认草拟前缀复用最多三行旧 MTP K/V，并用目标隐藏状态重算末尾至少两行。中文、代码、英文分别少重算 234、266、272 行，三份 1024-token 输出均逐 token 一致。验证由 213/241/271 增至 216/245/282 轮，解码为 48.90 → 48.72、34.01 → 33.51、38.89 → 38.10 token/s。每轮 MTP 起始及首候选读回约 2.47/3.02/2.55 ms，复用后基本持平。中文全程 nominal，另两组全程 fair。峰值最高约 42.39 GiB，可用内存最低 51%，无新增换出或交换。复用方案停在实验阶段。见 `runs/frontier-prime-reuse-retest.log`。

在固定草拟轨迹上，以路径概率至少 0.35 且 8K 草拟词表内首选概率至少 0.90 为条件，筛出中文、代码、英文的 658/813/713 个可省补扫的位置，分别漏掉原已提交路径上的 0/1/4 个正确首选。回放基于原先打分的前八名与此前目标预测，候选在窄词表中的概率使用下界。原版 BF16 词表权重微基准中，将补扫判断保留在 GPU 上，全 32K 打分的 0.68–0.73 ms 在跳过补扫时降至 0.46–0.52 ms，需要补扫时为 0.72–0.79 ms。补扫结果与单次全表的分数及前四名逐位一致。微基准计时阶段峰值低于 0.73 GiB，可用内存最低 71%，无新增换出或交换。见 `runs/frontier-conditional-vocab-replay.log`、`runs/frontier-conditional-vocab-probe.log` 和 `runs/frontier-conditional-vocab-device-retest.log`。

8K 单模型交替对照中，中文、代码、英文分别为 48.92 → 49.77、44.42 → 43.67、40.33 → 40.38 token/s，验证轮数为 213/241/271 → 213/243/275。融合保留行写回后，按提示词内 8K–32K ID 的数量选择 8K 或 16K，三组解码为 48.78 → 49.89、44.39 → 43.89、36.90 → 36.89 token/s，验证轮数为 213/241/271 → 213/241/276。两次对照三组均逐 token 一致。融合对照中文为 nominal，代码和英文为 fair。双分段表驻留的对照峰值最高 42.84 GiB，可用内存最低 51%，无新增换出或交换。服务仅为符合提示词条件的请求启用 8K，其他请求使用原 32K 草拟词表。见 `runs/frontier-conditional-vocab-ab.log` 和 `runs/frontier-conditional-vocab-fused-retest.log`。

正式 HTTP 入口复测 4096 输入 / 1024 输出。中文、代码、英文的首次解码为 47.92、43.28、39.82 token/s，分别验证 213、241、271 轮。中文在同一驻留服务复测为 49.26 token/s，首 token 13.50 秒，热状态在请求内由 nominal 转为 fair。两次中文均有 786 次分段评分，其中 214 次补扫。四次输出哈希均与历史基线一致，峰值约 42.48 GiB，系统可用内存最低 51%，无新增换出或交换。补扫及保留行写回通过 0、1、50 行的逐位校验。模拟额外 128 MiB 预留失败时，完整草拟词表正常进入预填充。短请求、非法调度、超上下文、进程互斥与健康检查通过。见 `runs/http-short-draft-head.log` 和 `runs/http-short-draft-head-warm.log`。

在已记录的宽批中，中文、代码、英文分别有 27/39/18 次扩宽未多确认 token，若事先知道这些结果，按中位面板耗时估算可省 0.51/0.72/0.33 秒。用八位置时可见的分叉拓扑、草拟队列和已验证历史拟合扩宽收益，整组留出后在统一 0.65 阈值下的预计净收益仅为 -1/50/26 ms。固定轨迹估算保留原草拟决策，宽批策略沿用现有实现。见 `runs/frontier-width-holdout.log`。

词表预取在同一个 Metal kernel 中把旧 BF16 行和新解码行写入目标顺序，省去先合并再重排的缓冲。候选读回与完整目标首选在同一次 GPU 同步内完成，下一轮的词表行在首次评分时求值。整模交替对照中，中文、代码、英文解码为 49.18 → 50.32、44.25 → 44.06、40.38 → 41.31 token/s，均复现 213/241/271 轮、排序和词表统计，1024 个输出逐 token 一致。英语结束时转为 fair，其他两组保持 nominal。单模型共同峰值最高 42.61 GiB，可用内存最低 51%，无新增换出或交换。见 `runs/frontier-fused-prefetch-ab.log`。

将主模型注意力 K/V 按 128 个归约分区连续存放。6–16 位置的线性与分叉输出逐位一致，四组只读耗时基本持平，计入新 K/V 写入后每层增加约 0.079–0.087 ms。该布局停在合成微基准，峰值低于 0.11 GiB，可用内存最低 74%。见 `runs/gqa-partition-kv-layout.log`。

融合词表预取上线后，默认 HTTP 的 4096/1024 中文、代码、英文解码分别为 49.57、44.23、41.02 token/s，同一驻留进程第二次中文为 49.69 token/s。首 token 分别为 13.49、13.50、13.51、13.45 秒。输出哈希、213/241/271/213 轮验证、草拟宽度和预取计数均与上线前一致。四次请求峰值约 42.48 GiB，可用内存最低 51%，无新增换出或交换。冷却后开始的四次请求均为 nominal，代码和第二次中文结束时为 fair。正式行轮换覆盖全新、全部复用和混合三类写出，短请求、超上下文、非法调度、互斥和健康检查均通过。见 `runs/http-fused-prefetch.log`。

分支接受后，仅保存需要搬动的 K/V 槽位，在下一次注意力批次与新增 K/V 一次写入。整模对照通过接续 1/3/8 个位置的隐藏状态、已用 K/V、卷积和 DeltaNet 状态逐位校验。中文解码 50.01 → 50.26 token/s，同场再提前排队 MTP 为 49.94 token/s。代码首次交替为 43.60 → 43.03，两路预热 96 个 token 后为 34.38 → 34.40。英文在 fair 下为 31.29 → 31.52。三组保持原输出和 213/241/271 轮验证。共同峰值最高 43.21 GiB，可用内存最低 50%，无新增换出或交换。服务上线 K/V 合并写入。见 `runs/frontier-cache-commit-pipeline-retest.log`、`runs/frontier-cache-commit-warm-code.log` 和 `runs/frontier-kv-commit-route-replay.log`。

K/V 合并写入的正式 HTTP 复测中，中文两次为 49.44/49.68 token/s，代码为 44.21，英文为 40.93。三种请求分别合并 46/44/76 轮写入，输出哈希和 213/241/271 轮验证复现。正式实现对容量 512 → 1024 的扩容边界以及接续 1/2/3/8 个位置通过逐位核对。四次请求峰值约 42.48 GiB，可用内存最低 51%，全程 nominal，无新增换出或交换，健康检查通过。见 `runs/http-kv-commit-retest.log`。

按实测耗时，中文、代码、英文各轮平均确认 4.80/4.24/3.77 个 token。若验证轮数和每轮主模型耗时固定，单计目标校验的条件吞吐为 58.01/51.86/47.68 token/s。达到 100 token/s 在现有每轮总成本下对应约 105/106/110 次验证，实际为 213/241/271 次。该预算用来比较减少验证次数和降低单轮成本的影响。见 `runs/frontier-verified-throughput-budget.log`。

预留 K/V 也试验了使用已知 CPU 偏移的静态切片写入。原始缓存和新缓存通过逐位核对，覆盖 1/8/16 位置和缓存扩容。预热后的四组微基准单次约少用 1–2 µs。两路各预热 96 个 token 后，中文由 49.38 升至 50.41 token/s，代码由 44.90 降至 44.10，原来分别为 213 和 241 轮。两组输出逐 token 一致，共同峰值不超过 42.61 GiB，可用内存最低 51%，无新增换出或交换。静态偏移保持在实验阶段。见 `runs/frontier-static-kv-append.log`、`runs/frontier-static-kv-append-warm.log`。[MLX 静态切片实现](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/indexing.cpp#L656) [MLX 动态切片实现](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/gpu/primitives.cpp#L101)

对同轮已评分旁支做后缀对齐，并尝试在既有节点预算内复用其续写。即使提前知道真实路径，中文、代码、英文也只分别多覆盖 2/2/4 个位置，按当时概率选取的净增量为 0/−1/0。该试验使用固定轨迹。见 `runs/frontier-same-round-donors.log`。

从现行 BF16 HTTP 调度复现三组输出，采集每轮已确认隐藏状态。用先前位置的目标与 MTP 隐状态均方根标准化差值做指数移动平均，并在更深的 MTP 输入上加权反馈。每组 128 个八步回放窗口中，中文、代码、英文的前四名命中分别为 816 → 807、864 → 842、846 → 831。连续前四名命中均值分别为 4.60 → 4.55、5.34 → 5.16、4.66 → 4.70。回放每组峰值低于 40.95 GiB，采集峰值约 42.48 GiB，系统可用内存最低 51%，无新增换出或交换。反馈没有进入正式调度。见 `runs/mtp-causal-hidden-feedback.log`。

把全注意力层的提示词 K/V 留作只读缓冲，只向生成后缀写入新位置和选中的旁支。逻辑归约按原顺序读取两段 BF16 缓冲，四组 6/8/11/16 位置的注意力输出与新缓存逐位一致。计入缓存更新，四组微基准各少用约 24–29 µs。整模交替对照中，中文、代码、英文分别为 49.53 → 50.22、44.52 → 43.88、34.64 → 35.11 token/s。代码两路约 0.34 秒的差异集中在前 256 个输出 token。三组输出逐 token 一致，验证轮数和草拟宽度相同。峰值最高 42.40 GiB，可用内存最低 51%，无新增换出或交换。见 `runs/frontier-immutable-prompt-kv-ab.log`。

部署后，四次默认 HTTP 请求均在 16 层全注意力启用分段 K/V，复现原输出哈希、213/241/271/213 轮验证和分叉合并次数。中文首轮和复测分别为 49.07、49.71 token/s，代码为 44.28，英文为 40.95。请求峰值约 42.48 GiB；代码和英文的解码峰值比上一版均降低约 16 MiB。四次可用内存最低 51%，无新增换出或交换。正式内核及单 token 回退通过逐位检查，短请求、非法调度、超上下文、互斥和健康检查通过。见 `runs/http-immutable-prompt-kv.log`。

MTP 草拟层也保留 4095 个位置的只读 K/V，分叉只复制可写后缀。单查询 Metal 内核让相邻两个查询头共享 K/V 读取，并沿用原 BF16 归约顺序。4096 至 5119 的五个历史长度与原单查询注意力逐位一致，缓存更新加注意力每次少用 25–42 µs。单份模型交替对照的中文、代码、英文分别为 49.38 → 50.46、44.67 → 44.14、40.42 → 41.11 token/s，两路总耗时存在首次建图顺序差异。三组草拟阶段分别省 0.10、0.04、0.11 秒，1024 个输出 token 和验证轮数逐组一致。共同峰值最高 42.39 GiB，可用内存最低 51%，无新增换出或交换。见 `runs/mtp-prompt-kv-probe.log` 和 `runs/frontier-mtp-prompt-kv-ab.log`。

默认 HTTP 的中文首轮和复测为 49.08、49.88 token/s，代码为 44.32，英文为 41.12。四次都记录了 4095 个只读 MTP 位置，输出哈希、213/241/271/213 轮验证及草拟统计均复现。正式内核逐位复现原注意力，覆盖后缀从 512 扩至 1024 位置及多查询回退。请求峰值约 42.48 GiB，解码峰值约 41.66 GiB，系统可用内存最低 51%，无新增换出或交换。短请求、超上下文、进程锁和健康检查通过。见 `runs/http-mtp-prompt-kv.log`。

从五层原始 BF16 FFN 权重的有界行样本中测量连续七个指数值的覆盖率。以 3 位编码这七个值、用第八个码保留需按原 8 位存储的例外，例外占 2.12%–3.13%，理想载荷约为原权重的 69.81%–70.32%。估算尚未计入分块寻址、对齐和解码成本。此项仅使用 CPU 读取权重样本。见 `runs/bf16-individual-exponent-audit.log`。

在第 23 层真实 FFN down 矩阵上验证这一布局。GPU 通过 SIMD 前缀和寻址例外缓冲，覆盖全部常规指数和随机 BF16 位型，并逐位恢复整层权重与 6/8/11/16 位置的 FP32 部分和。载荷由 129.50 降至 120.24 MiB，单层投影增加 0.044–0.077 ms。峰值 1.49 GiB，服务恢复后健康检查通过。这一布局保留在实验中。见 `runs/bf16-three-bit-exponent-probe.log`。[Apple Metal Shading Language 规范](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)

在主模型 FFN down 的末尾合并归约与 BF16 残差写回，保留 FP32 部分和的归约顺序和两次 BF16 舍入。6/8/11/16 位置的合成与第 23 层真实权重测试均逐位一致。仅测归约时减少约 5–9 µs，计入投影后，与编译过的基线路径相比基本持平或稍慢。另一路把所有分片放进同一线程组，省去单层 3.98–10.63 MiB 的 FP32 中间缓冲，但 16 SIMD 组版本每层增加 0.024–0.045 ms。两项均留在微基准，峰值 1.54 GiB 以下，结束后服务健康。见 `runs/down-residual-epilogue-probe.log` 和 `runs/down-inline-residual-probe.log`。[MLX 0.32.4 Metal 分片归约](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/reduction/reduce_col.h#L190)

词表预取在物理词表顺序过滤提示词与固定草拟区，只转换选出的 16 个候选，再以 FP32 计算分差。编译并用 BF16 分段选择后，6/8/11/16 位置的 28 组合成样本逐位一致，四组提示提取约快 19%–23%。单份模型共用前缀，双路各预热 96 个输出 token 后，中文、代码、英文分别为 50.62 → 50.61、44.87 → 44.83、41.45 → 41.58 token/s。三组输出及验证轮数一致，共同峰值最高 42.37 GiB，可用内存最低 51%，无新增换出或交换。正式服务保留现行词表提示提取。见 `runs/frontier-compiled-tile-hints-ab.log`。

按实际四 tile 布局抽样五层 FFN，估算用一个 lane 位图和每个异常 lane 两个原始指数槽替代逐 lane 前缀和。每 lane 有超过两个例外时，整包使用原布局。样本中 1.68%–7.42% 的包需回退，计入描述符和原始指数后，预计比同样本上的现有单矩阵无损布局少 3.39%–4.85% 字节。当前结果来自 CPU 载荷估算。见 `runs/bf16-bounded-escape-lanes-audit.log`。

在独立启动的 HTTP 服务上测试完整解码路径的预热。候选在默认 80 GiB 内存预算内执行十种验证形状，额外耗时 0.97 秒，中文 4096/1024 的首轮与重复请求分别为 49.96、50.02 token/s，两次全程 nominal。初始对照为 38.03 和 49.56 token/s，热状态在请求中进入或保持 fair。恢复正式服务后，首次请求在全程 nominal 下为 49.84 token/s；与候选首轮相比只差约 0.047 秒。所有请求复现输出哈希和 213 轮验证，峰值约 42.48 GiB，可用内存最低 51%，无新增换出或交换。候选所在进程的启动全程峰值约 46.10 GiB。正式服务保留原启动流程并通过健康检查。见 `runs/http-decode-startup-warm-ab.log` 和 `runs/http-decode-startup-warm-baseline-retest.log`。

实现带 lane 位图和两字节例外记录的三位指数布局。第 23、61 层 FFN down 均全量逐位还原，6/8/11/16 位置的 FP32 部分和逐位一致。真实载荷分别减少 2.91% 和 1.28%，较快的向量实现仍每层增加约 0.06–0.07 ms。两层依次载入，测试峰值 1.49 GiB，系统可用内存最低 72%，恢复服务后健康检查通过。见 `runs/bf16-bounded-escape-probe.log`。

主模型短批全注意力的 QKV 投影和输出投影至 FFN 分别编译，沿用现有 K/V 更新和逻辑位置归约。两层真实 BF16 权重在 6/8/11/16 位置通过隐藏状态和缓存逐位核对。微基准每层每轮建图少用约 10–12 µs，含执行少用约 6–21 µs。单份模型以同一 4K 缓存交替生成，中文、代码、英文分别为 50.60 → 50.82、44.96 → 45.00、41.21 → 41.37 token/s，输出和 213/241/271 轮验证完全一致。峰值最高 42.37 GiB，可用内存最低 51%，无新增换出或交换。英文结束时进入 fair。见 `runs/compiled-attention-sides-probe.log` 和 `runs/compiled-attention-sides-ab.log`。

对每个当前 token，枚举提示词与已经生成位置中同 token 的所有后继，允许提前知道哪条已见续写能多确认 token，并与已记录的原生验证路径共同回放。中文、代码、英文的固定轨迹最优轮数分别由 213/243/277 降至 211/240/277。回放允许主动缩短单轮，忽略追加分叉的开销，因此结果只给出这类精确复制策略在固定轨迹上的上限。见 `runs/frontier-same-token-source-ceiling.log`。

正式编译路径通过 1/2/3/5/6/8/11/16 位置的隐藏状态与 K/V 逐位核对，包含单位置回退、后缀扩容和旁支合并。五次默认 HTTP 请求的中文首轮、代码、英文和两次中文复测分别为 49.29、43.55、39.76、48.75、48.62 token/s，全部复现输出哈希和验证轮数。随后在同一驻留 HTTP 模型上按原路径、编译、编译、原路径顺序复测中文，四次全程 nominal，分别为 48.88、49.05、49.88、49.85 token/s。最后两次主模型验证时间为 17.60、17.62 秒，HTTP 收益小于离线交替对照。保留编译路径，恢复正式服务并通过健康检查。九次请求峰值约 42.48 GiB，可用内存最低 51%，请求期间无新增换出或交换。见 `runs/http-compiled-attention-sides.log` 和 `runs/http-compiled-attention-resident-ab.log`。

以运行时有效长度读取固定容量的 K/V 后缀，并编译完整全注意力层。两层真实 BF16 权重通过首次分配、扩容、线性和分叉以及 2/3 行非连续回滚的逐位核对。单层建图少用约 11–16 µs。整模交替对照的中文、代码、英文分别为 50.47 → 48.73、44.87 → 44.53、39.80 → 39.85 token/s，三组输出和验证轮数一致，峰值最高 42.37 GiB，可用内存最低 51%，无新增换出或交换。旧轨迹中，完整方案可见的宽度、拓扑、容量及回滚行数共形成 49–53 种组合，其中 37–43 种在前 96 个输出 token 之后首次出现。这些组合数未计入不同层的权重编码形状。见 `runs/compiled-capacity-attention-probe.log`、`runs/compiled-capacity-attention-ab.log` 和 `runs/compiled-capacity-shape-audit.log`。

在存量代码负载的原生 MTP 固定轨迹上，预知正确候选的排名并计入排名更高的备选节点。保持最多 16 个验证位置，将可保存的分叉增至四个时，估计最少轮数由 182 降至 175，基于同场历史批宽成本的乐观吞吐由 62.16 升至 63.10 token/s。放宽至 32 位置并按 16 位置成本计费也只估到 66.22 token/s。这是已知未来的 CPU 回放。见 `runs/mtp-fork-count-ceiling.log`。

只编译无回滚的 8/16 位置批时，轨迹可见组合降至 8–9 种。整模中的中文、代码、英文分别为 35.08 → 35.11、21.56 → 21.55、15.59 → 15.57 token/s。每个对照的两路都在 fair，运行期间连基线也明显降速。这组数据只用于同场比较。三组输出及验证轮数一致，峰值最高 42.37 GiB，可用内存最低 51%，请求期间无新增换出或交换。见 `runs/compiled-capacity-regular-ab.log`。

将新 K/V 写入和上一轮旁支重排融合成一个 Metal kernel，固定传入 16 槽路由，实际回滚行数作为运行时数据。旧轨迹中，8/16 位置的完整编译图可覆盖约 86%–89% 的轮次，组合数仍为 8–9 种。两层真实 BF16 权重的 12 组隐藏状态和 K/V 均逐位一致，单层平均少用约 21 µs。整模的中文、代码、英文分别为 50.53 → 50.87、44.99 → 45.00、41.29 → 41.45 token/s，输出和 213/241/271 轮验证一致，峰值最高 42.37 GiB，可用内存最低 51%，无新增换出或交换。中文全程 nominal，代码结束时 fair，英文全程 fair。见 `runs/compiled-fused-kv-attention-probe.log` 和 `runs/compiled-fused-kv-attention-ab.log`。

融合写入上线后，默认 HTTP 的中文冷请求为 48.91 token/s，两次驻留复测分别为 50.16 和 50.14 token/s，端到端分别耗时 33.93 和 33.88 秒。代码和英文为 44.59 和 40.97 token/s。所有请求的 1024 个输出与验证轮数复现，统一内存峰值约 42.48 GiB，解码峰值最高约 41.67 GiB，可用内存最低 51%，无新增换出或交换。正式路径首次分配、后缀扩容、非连续 2/3/8 行回滚及单位置回退逐位一致。短请求、超上下文、进程互斥及健康检查均通过。见 `runs/http-joint-attention-kv.log`。

在当前分段 K/V 路径上，把连续三层 DeltaNet 及随后的全注意力层编入同一张图。四种 8/16 位置的一分叉、两分叉形状在真实 BF16 权重上逐位复现输出、FP32 递推与主链或旁支提交。四层图建图少用约 9–10 µs，微基准峰值 4.88 GiB。单份模型的中文、代码和英文分别为 49.43 → 49.64、44.09 → 43.91、38.78 → 38.80 token/s。三组逐 token 一致，均复现 213/241/271 轮，峰值最高 42.37 GiB，可用内存最低 51%，无新增换出或交换。四层图保留在实验中。见 `runs/compiled-hybrid-block-probe.log` 和 `runs/compiled-hybrid-block-ab.log`。

尝试在单路径 MTP 分数读回前用设备端首选再走一步，只有后续调度确实选择同一路径的单行投影才复用。代码负载尝试 385 次，复用 318 次，草拟为 3.305 → 3.357 秒，解码为 44.25 → 44.01 token/s，仍逐 token 复现 241 轮。整模峰值 42.01 GiB，可用内存最低 51%，无新增换出或交换。两路开始时 nominal，结束时 fair。提前一步未上线。见 `runs/frontier-device-lookahead-ab.log`。

用当前部署配置测量主模型 40.09 GB 权重载荷。一次完整读取中位数为 57.11 ms，有效吞吐约 702 GB/s，其中 FFN 单独读取为 36.24 ms。用中文原输出推进前缀，在后缀长度 0、512、896 时测试固定树形、完整词表及预取提示逻辑。8 位置验证为 74.77–75.96 ms，16 位置为 93.19–94.16 ms，重复预测一致。加上现行 HTTP 草拟和提交的每轮实测时间，100 token/s 对应 8 位置轮平均推进 8.81–8.90 个 token，或 16 位置轮推进 10.69–10.78 个 token。现行中文、代码和英文平均推进 4.80、4.24、3.77 个 token。读取计时用作成本参考。这组测量全程 nominal，峰值 41.65 GiB，可用内存最低 51%，无新增换出或交换，服务恢复后健康。见 `runs/current-verifier-roofline.log`。

复现中文、代码、英文 4096/1024 请求的全部输出 token 及 213/241/271 轮验证。沿已确认输出做 12 步 MTP 回放，分别换入当前位置的真实目标隐藏状态和此前由真实隐藏状态生成的 MTP K/V。单独换隐藏状态补回 76/176/172 个原先未进前四名的位置，单独换 K/V 补回 13/24/24 个。回放短词表另有 135/13/27 个位置缺少正确词条。以每个窗口的已确认隐藏状态为锚，将 12.5% 混入后续递推，前四名连续覆盖长度从 6.10/6.05/5.23 降至 5.44/5.51/4.97，此方案保留在实验中。真实状态对照使用当轮在线草拟时尚未获得的状态，连续覆盖长度只用于诊断。模型峰值 42.48 GiB，回放峰值 40.88 GiB，系统可用内存最低 51%，未新增换出或交换。见 `runs/mtp-recurrence-isolation-retest.log`。

在现行 4096/1024 验证轨迹比较第 16/32/48 层的全词表读出、前 16/32 层接最后 4 层，以及用上轮末尾已确认 token 校正第 32 层的读出。前 32 层接最后 4 层的覆盖最高，中文、代码、英文各有 343/272/342 个位置的正确 token 排第一。在已有草稿分支却漏掉正确 token 的 152/176/192 个位置中，分别补中首选 12/15/15 次，前四名覆盖 33/38/55 次。尾部四层使用每轮开始时的真实历史缓存。三组同形状计时分别为 62.19/50.67/73.48 ms，约占同场完整验证耗时的 57%/57%/56%，热状态均为 fair。完整 64 层输出和 213/241/271 轮验证复现，试验峰值 43.04 GiB，可用内存最低 51%，无新增换出或交换，服务恢复后通过健康检查。部分层读出留作实验。见 `runs/target-sparse-layer-readout.log`。 [Draft & Verify](https://aclanthology.org/2024.acl-long.607/)

对较早的完整验证轨迹做 CPU 回放，以最新确认的 token 替换上轮分叉的首个错误，然后按已提交前缀和当时分数选择剩余草稿作并行递推初值。中文、代码、英文分别可提供 451/583/465 个初始槽位，其中 77/99/65 个与真实 token 对齐，连续前缀覆盖 49/82/56 个位置。即使用后来才知道的输出选择上轮分叉，连续覆盖也只有 63/86/62 个位置。结果用于评估上轮错支对新一轮的初值质量。见 `runs/mtp-span-prior-seed-audit.log`。

从中文、代码、英文正式验证路径上的草拟节点收集 638/653/684 对草拟与目标隐藏状态。每个 12 步回放窗口只用此前完成验证的样本，按递推深度分两组，各最多保留 64 对，按隐藏状态相似度估计残差。留出区间内，去掉组均值后的近邻残差将前四名连续覆盖从 6.42/6.08/5.22 变为 5.92/6.11/5.20，三组首选连续覆盖均下降。保留组均值的残差覆盖下降更多。全量输出和 213/241/271 轮原验证复现，回放峰值低于 40.90 GiB，无新增换出或交换。此方案保留在实验中。见 `runs/mtp-verified-neighbor-replay.log`。

草拟词表在同一次 Metal 扫描中计算各组前四名及归一化所需的最大分数和指数和，再用一次归约选出全局候选并计算其 log 概率。对 1–3 行、32768–33021 个 BF16 分数的微基准，前四名归约从 0.286–0.298 ms 缩至 0.213–0.217 ms，候选顺序完全一致，log 概率最大绝对差为 1.91×10⁻⁶。覆盖了分数并列和分段草拟词表中被屏蔽的分数。同模型交替对照在 fair 热状态下分别把中文、代码、英文的草拟阶段缩短 0.139/0.197/0.203 s，整段解码缩短 0.229/0.226/0.200 s。三组 1024 个输出、213/241/271 轮验证和调度计数均一致。交替对照的解码峰值 42.37 GiB，可用内存最低 51%，换出与交换计数不变。见 `runs/frontier-joint-rank-mass.log` 和 `runs/frontier-joint-rank-mass-ab.log`。

合并归约已部署。集成检查覆盖 1–3 行、32768–36864 词表以及小词表、前八名和超大词表回退。三组默认 HTTP 4096/1024 请求分别为 49.06/44.02/41.34 token/s，复现原文本哈希、验证轮数及草拟候选数。三个请求启动时均为 nominal，英文请求结束时为 fair。请求峰值 42.48 GiB，可用内存最低 51%，无新增换出或交换。短请求回退、非法参数、超上下文、进程互斥和健康检查通过。见 `runs/http-frontier-joint-rank-mass.log`。

在同一驻留服务上复测两次中文 4096/1024 热请求，解码为 50.20 和 50.24 token/s，端到端为 33.90 和 33.92 s，草拟阶段均约 2.68 s。两次启动时均为 nominal，第二次结束时为 fair。输出哈希、213 次验证及草拟候选数一致，两次峰值均为 42.48 GiB，可用内存最低 51%，无新增换出或交换。见 `runs/http-frontier-joint-rank-mass-warm.log`。

在真实 32K BF16 草拟权重的投影写回中直接计算每组前四名和指数和。用各组首选确定最多四个参与最终排名的组。微基准覆盖 1–3 行、0/256/1024 额外词表行、前三或前四名，候选逐项一致，log 概率最大绝对差 9.54×10⁻⁷，耗时仅减少 0.1–2.4 µs。另测分段词表省去未启用补扫时的 24K 掩码分数写回，直接对实际行归约。候选一致，最大 log 概率差 1.91×10⁻⁶，跳过补扫时少 4.0–8.0 µs，GPU 自行判断补扫的一组快约 1.7 µs、另一组慢约 2.1 µs。两条路径保留在微基准。只加载词表权重，峰值最高 2.87 GiB，可用内存最低 74%，无新增换出或交换，正式 BF16 服务恢复后通过健康检查。见 `runs/frontier-projected-rank4.log` 和 `runs/frontier-compact-conditional-rank.log`。

用最新正式 HTTP 的 1023 个解码 token、目标验证和提交耗时估算。在保持已测验证轨迹的条件下，把草拟耗时设为零，中文、代码、英文的估算解码吞吐为 57.94/51.08/47.56 token/s。按各自当前的平均验证及提交成本，100 token/s 最多容许 123/123/128 轮，对应实测 213/241/271 轮。这是当前轨迹的条件估算，用来衡量减少验证轮数和单轮成本的必要幅度。见 `runs/frontier-draft-free-ceiling.log`。

用四层真实 BF16 权重测试现有无损权重包的指数还原。将每对指数放进一个 32 位字的高低半字，一次完成加偏置和移位，长度及排布均沿用原权重包。完整回读、原样回退、8/11/16 节点的 gate/up 输出和 down 的 FP32 部分和均逐位一致。11/16 节点的 gate/up 由约 0.600 ms 降至 0.588/0.589 ms，down 由 0.360/0.361 ms 降至 0.351/0.352 ms。8 节点 gate/up 增加约 6 µs，单 lane 读描述符再广播未优于宽批直接还原。微基准全程 nominal，峰值 4.13 GiB，可用内存最低 71%，无新增换出或交换。见 `runs/verify-packed-exp-decode.log`。 [Metal SIMD 组](https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups)

只在 9–16 位置启用上述解码，在一份原版 BF16 模型和相同 4K 缓存上测量。生成后缀长度 0/512 的 11 位置线性验证各省 1.41/1.32 ms，16 位置树验证各省 1.70/1.66 ms，两种 8 位置树的配对差小于 0.03 ms。六种整模状态的隐藏值、预测、K/V 和 DeltaNet 提交状态逐位一致。完整中文 4096/1024 交替对照共有 70 轮宽批，解码为 50.61 → 50.16 token/s，目标用时为 17.34 → 17.55 s。两路都复现 213 轮、输出和草拟宽度。峰值 42.08 GiB，可用内存最低 51%，热状态 nominal，无新增换出或交换。见 `runs/verify-packed-exp-model-ab.log`。

双路预热后，从同一真实中文请求复制 54 个宽批的入轮验证状态，交替执行同一批 token 和树路由。配对中位数每轮节省 1.77 ms，15 个带待提交 K/V 的宽批也节省约 1.77 ms。25 种形状首次出现时逐位核对隐藏状态、预测和提交缓存。再次完整交替生成时，两路均为 fair，解码 43.59 → 43.83 token/s，目标用时 20.44 → 20.29 s，全部输出及 213 轮验证一致。真实请求观测峰值 42.48 GiB，可用内存最低 51%。两个计时区间内换出和交换计数不变，区间之间主机换出计数增一。见 `runs/verify-packed-exp-live-shadow.log`。

宽批 BF16 指数还原接入正式服务，沿用启动图预热。集成检查覆盖 6/8/9/11/13/16 个位置。正式 HTTP 中文、代码、英文 4096/1024 请求复现原输出哈希、213/241/271 次验证和草拟统计。三组均为 fair，原始解码计时为 35.68/26.47/20.98 token/s。三组峰值约 42.48 GiB，可用内存最低 51%，请求期间无新增换出或交换。短请求、非法参数、超上下文、进程互斥及健康检查均通过。见 `runs/http-verify-packed-exp.log`。

上线后空闲等待 120 秒，热状态未达到连续三次 nominal 的计时条件。本次未发起热请求，BF16 服务通过健康检查。见 `runs/http-verify-packed-exp-warm.log`。

把主模型配对 BF16 包的两个指数基准和载荷索引合到一个 32 位描述符，仍从既有的压缩与原样载荷读取权重。10 个采样矩阵的位模式、行优先回读和 8/11/16 位置投影逐位一致，指数或索引超范围时回退原格式。整模检查 64 个 gate/up、48 个 DeltaNet 输入和 16 个注意力 QKV，128 个均可转换，描述符常驻空间共省 123.75 MiB。借用同一份其余权重做对照，64-token 预填充与两个后缀上的 8/16 位置树验证均逐位匹配隐藏状态、预测和提交缓存。中文 4096/1024 同场复测为 47.80 → 48.36 token/s，验证耗时 18.39 → 18.11 s，全部输出、213 轮验证及草拟统计一致。双路并存时峰值 42.20 GiB，可用内存最低 51%，热状态 nominal，无新增换出或交换。见 `runs/verify-packed-pair-descriptors.log` 和 `runs/verify-pair-descriptor-full-ab.log`。

首次集成的权重回读仍走了原格式入口，逐位检查失败后恢复服务。修正后 6/8/11/16 个位置均通过检查并上线。正式 HTTP 中文、代码、英文的 4096/1024 解码为 46.07/42.54/39.09 token/s，复现原文本哈希、213/241/271 次验证和草拟统计。三个请求启动时均为 nominal，英文请求结束时为 fair。请求峰值 42.36 GiB，可用内存最低 51%，三组请求均无新增换出或交换。短请求、非法参数、超上下文、进程互斥及健康检查均通过。见 `runs/http-paired-compact-descriptor-retest.log`。

按早期保存的中文、代码、英文验证轨迹，把每轮正确续写首次未进入验证树的位置按深度归档。三组分别有 64/111/123 个正确 token 已在 MTP 前四名，其中 47/86/95 个发生在根节点至深度 3。深度 4 及之后分别有 47/29/25 个末端尚未评分。此处使用之前收集的路径和候选轨迹。见 `runs/frontier-first-break-depth.log`。

在当前 BF16 模型上复现三组 4096/1024 输出和 213/241/271 轮验证，记录已确认隐藏状态并做 12 步回放。仅改变草拟打分前的 MTP FFN 残差，比较原路与 0.5/1.5 倍残差，递推仍使用原隐藏状态。合批投影的原路分数逐位核对通过。三组备选首选均未补回原前四名之外的正确 token。两种备选的完整前四名分别可补 9/15/17 个独立位置，用备选首选替换第四名的实际回放补中 0 次，原连续覆盖保持一致。试验峰值 42.36 GiB，回放峰值 40.76 GiB，可用内存最低 50%，无新增换出或交换，正式服务已恢复。见 `runs/mtp-ffn-disagreement-replay.log`。

同机测试了 Inco AI Qwen3.8-27B-DFlash2 的原版 BF16 草稿权重，主模型沿用线上原版 BF16 验证器和词表布局。中文技术报告固定为 4096 输入、1024 输出。8 位置得到 19.33 token/s，326 轮验证，平均确认 3.14 token/轮。16 位置得到 12.49 token/s，324 轮验证，平均确认 3.16 token/轮。两组均与已保存的 MTP 输出逐 token 一致。8 位置全程为 nominal，16 位置由 nominal 降至 fair。加载草稿前释放 MTP 张量 0.79 GiB，加载峰值 45.98 GiB，预填充峰值 45.18 GiB，解码峰值最高 44.27 GiB，解码期间系统可用内存最低 50%，无新增换出或交换。实验结束后优化版 BF16 服务通过健康检查。见 `runs/dflash2-bf16-local-retry.log`。

用 DFlash2 的真实 BF16 logits 验证分块前 16 名归约。7 和 15 行的候选 ID 与顺序均和 MLX 全排序一致，耗时分别从 1.05 降至 0.43 ms、从 1.79 降至 0.46 ms，折算约占整轮耗时的 0.4% 和 0.5%。8 位置首次拒绝的 299 轮中，229 轮的目标 token 已进入草拟前 16 名。16 位置对应 242/322。数据支持优先研究如何让验证器一次接受更长的候选前缀。见 `runs/dflash2-bf16-local-retry.log`。

对同一中文 4096/1024 输出的 1023 个草拟起点逐一测量 DFlash2 宽度 16 的条件候选排名。先用已保存的正确续写按真实因果路径采集 BF16 隐藏状态，主模型的 1023 个预测全部与保存的 token 一致。回放允许预先知道正确候选和分轮位置。按候选分数顺序最多加两个分叉、每轮最多验证 16 个位置时，最少仍需 194 轮。若直接从每个位置的前 16 名挑出正确 token，并省去所有旁支成本，最少需 165 轮。按实测验证与草稿读权重的乐观估计，分别约为 51.75 和 60.84 token/s。100 token/s 对应 52.73 和 62.00 ms 的单轮预算。本机单独扫描主模型权重为 57.11 ms，草稿与共用词表的额外扫描估算为 7.89 ms。此回放用于评估候选覆盖与验证成本，实测解码速度见上一轮。试验峰值 45.98 GiB，系统可用内存最低 52%，无新增换出或交换，正式 BF16 服务已恢复。见 `runs/dflash2-bf16-branch-ranks.log` 和 `runs/dflash2-bf16-branch-ranks.json`。

64 个全注意力与 DeltaNet 输出矩阵改用无损 BF16 配对包，保留原有融合注意力和编译图。权重回读逐位一致，常驻载荷从 4.03 GB 降至 3.06 GB，少占 917.67 MiB。单份模型 A/B 在 4096 和 4608 上下文位置核对了 8/16 位置的隐藏状态、预测、旁支提交缓存和词表提示，均逐位一致。8 位置配对中位数每轮省 1.64–1.75 ms，16 位置省 0.94–0.97 ms。中文 4096/1024 同场完整解码为 51.33 → 51.64 token/s，两组输出、213 轮验证和草拟统计一致，共存峰值 44.83 GiB，热状态 nominal。见 `runs/verify-lossless-attention-outputs.log`。

部署后，实际 HTTP 中文、代码、英文的 4096/1024 解码为 51.49/45.45/42.15 token/s，包含预填充时的输出吞吐为 30.78/28.48/27.16 token/s。三组复现已保存的输出哈希和 213/241/271 轮验证，确认 64 个输出矩阵均通过无损路径。请求峰值 41.46 GiB，可用内存最低 52%，无新增换出或交换。三个请求启动时均为 nominal，英文请求结束时为 fair。短请求、非法草拟参数、上下文上限、Metal 互斥和健康检查均通过。见 `runs/http-lossless-attention-outputs.log`。

启用无损输出投影后，扫描目录中的主模型权重共 38.94 GB，读取中位数 55.86 ms，有效速率 697 GB/s。此轮热状态 fair，峰值 40.64 GiB，可用内存最低 52%，换出和交换计数不变，服务随后恢复。用最新 HTTP 的实际草拟与提交耗时计算，维持中文、代码、英文的 213/241/271 轮时，达到 100 token/s 的验证预算分别只有 34.94/29.11/25.41 ms，对应约 24.36/20.30/17.72 GB 的读量。将六组 BF16 样本的每权重经验熵外推到 checkpoint 的 401 个大投影矩阵，估计约 33.52 GB、48.09 ms。这是无损编码的估算情景。见 `runs/lossless-output-verifier-roofline.log` 和 `runs/bf16-100-tps-budget-after-output-pack.json`。

在保存的中文候选上，允许预知正确 token、忽略旁支占位，并在有记录的 MTP 起点接入 DFlash2 的前 16 名，最少需要 150 轮，单独 DFlash2 需要 165 轮。按轮选择 MTP、DFlash2 或拼接，只计主模型扫描、DFlash2 草稿读取和 MTP 实测草拟与提交，得到 9.66 秒的乐观估算。距 100 token/s 的 10.23 秒总预算仅余约 0.57 秒，平均约 3.8 ms/轮可用于全部未计开销。此回放用于筛选候选组合。见 `runs/mtp-dflash2-complement-budget.log` 和 `runs/bf16-100-tps-budget-after-output-pack.json`。

JetSpec++ 发布的 3.46 GB BF16 草稿检查了 SHA-256 和全部 58 个张量的形状及类型。用本机原版 BF16 主模型、真实已确认上下文、16 位置串行验证和固定的 4096 输入 / 1024 输出代码请求，对照初稿和一次 Jacobi 自修正。分别为 22.75 / 20.54 token/s、366 / 345 轮、2.80 / 2.97 token/轮。草拟约为 27.67 / 52.62 ms/轮。两组输出逐 token 一致，复现保存的 MTP 输出。这里测量的是共享一台 M2 Ultra 的真上下文串行路径；论文的 JetSpec++ 还包括适配器、上下文刷新和双 GPU 重叠。装载峰值 45.09 GiB，预填充峰值 43.92 GiB，解码峰值 43.17 GiB，可用内存最低 50%；两轮分别新增 36 / 78 次 pageout，交换计数无变化。正式 BF16 MTP 服务已恢复。见 `runs/jetspec-bf16-code-local.log`。 [JetSpec++ 论文](https://arxiv.org/html/2610.19792v1) · [BF16 草稿](https://huggingface.co/JetSpec/jetspec-plus-plus-qwen3p8-27b)

复用原生 MTP 输入投影的 FP32 lane 前缀。在已知单路 token 未命中时走原生 BF16 投影；两路或三路同时草拟且全部命中时，共用隐藏状态半矩阵的读取。真实 BF16 权重的两路和三路投影逐位一致，微基准由 0.379 / 0.379 降至 0.297 / 0.306 ms。中文 4096/1024 单模型交替对照绕过 360 次未命中，264 次多路投影中有 48 次全部命中。草拟阶段仅省 12.6 ms，完整解码 51.09 → 51.04 token/s，输出和 213 次验证一致。峰值 41.08 GiB，可用内存最低 51%，热状态 nominal，无新增换出或交换，线上 BF16 服务健康。见 `runs/mtp-known-lane-dispatch-ab.log`。

验证词表的预取提示改用实验性 Metal 分块筛选，每 256 个词表块保留前 16 个，再做全局归并。对 6/8/11/16 个验证位置，56 组 BF16 样本的候选、并列顺序与 FP32 分差逐位一致，包含全屏蔽、正负零、无穷和 NaN。微基准从 0.230–0.331 ms 降至 0.129–0.145 ms。单份模型交替完成 4096/1024 对照，中文、代码、英文分别为 50.43 → 50.69、45.15 → 45.10、40.83 → 41.08 token/s，输出、213/241/271 轮验证与预取计数均一致。完整解码差异小于 0.7%，分块路径仅用于基准。峰值 41.36 GiB，可用统一内存最低 51%，换出和交换计数无变化。见 `runs/frontier-bounded-tile-hints-micro.log` 和 `runs/frontier-bounded-tile-hints-ab.log`。

对四层真实 FFN gate/up 和 down 测试另一种无损 BF16 包，原样保存低字节，用四位保存符号与高指数差，超范围包原样回退。全部 65,536 种 BF16 位型、随机及压缩/回退混合包逐位回读通过，4/6/8/11/16 位置的 FP32 down 部分和、BF16 SwiGLU 输出逐位一致。低字节方案在采样矩阵上增加 0.34%–0.92% 载荷。8 位置的 gate/up 和 down 由 0.4725/0.2632 增到 0.4770/0.2646 ms。最新 HTTP 三组负载各有 61.0%/67.2%/76.8% 的验证轮为 8 位置；按实测批宽分布、64 层与同档最近的微基准估算，整段解码增加约 0.05/0.07/0.09 s。候选仅用于微基准。首次试验在 gate/up 调用处停止，修正后完整复测峰值 5.11 GiB，可用统一内存最低 71%，本地原版 BF16 服务恢复并通过健康检查。见 `runs/bf16-verbatim-low-byte-projection-retest.log`。

对 4096-token 预填充作逐层计时，中文首 token 约 13.52 s，其中 FFN 约 7.81 s、DeltaNet 约 3.59 s、全注意力约 1.24 s、权重解包约 0.66 s。全注意力的输入/输出投影合批在三组提示词中保持隐藏状态和缓存逐位一致，完整预填充分别增加约 0.04/0.10/0.29 s。直接合并整个 DeltaNet 会改变小型 `a/b` 投影的 BF16 位模式；修正首次插桩定位后，三层测试确认将这两个投影保留在原 512-token 分块时，可以合并其余计算并保持逐位一致。见 `runs/prefill-attention-panels.log` 和 `runs/prefill-delta-projection-panels-retest.log`。

在没有分段计时同步的单模型对照中，选择两组各 2048 token 的 DeltaNet 路径。中文首 token 两侧对照分别为 13.50 和 13.40 s，新路径为 13.15 s，全部 4096 个隐藏状态、64 层缓存与 MTP 缓存逐位一致。正式 HTTP 的中文、代码、英文 4096/1024 首 token 为 13.17/13.49/14.50 s，解码为 51.31/44.80/38.26 token/s。全程热状态 fair，三组输出哈希、213/241/271 轮验证及草拟计数与原版 BF16 服务一致。请求峰值 41.50 GiB，可用统一内存最低 52%，无新增换出或交换。短请求、非法调度、上下文上限、Metal 互斥和健康检查均通过。见 `runs/prefill-delta-direct-panels.log` 和 `runs/http-prefill-delta-panel-2048.log`。

用正式实现在单份驻留模型中交叉验证代码和英文 4K 首 token。通过同一个模型级开关选择原 512-token 路径和新的 2048-token 路径，调度顺序相反。代码为 16.32 → 16.24 s，英文为 17.45 → 16.54 s，四次均为 fair。两组的完整隐藏状态、全部目标缓存和 MTP 缓存逐位一致。对照峰值 41.71 GiB，可用统一内存最低 69%，无新增换出或交换。见 `runs/prefill-delta-panel-production-ab.log`。

采集原版 BF16 MTP 在中文、代码、英文正式 4096/1024 调度中的 1535/1695/1814 个已打分节点，包括最后没有提交的分支。三组输出逐 token 复现，验证仍为 213/241/271 轮。每次用另外两类拟合 64 维中心化隐藏状态残差，在留出的整类负载上评估，固定每轮的活跃词表。离线 BF16 GEMM 与线上草拟打分的首选一致数为 1534/1695/1814，前四名集合一致数为 1318/1694/1814。保留原前三名、用修正首选替换第四名时，三组均没有补回漏选。即使把两份前四名都保留，已确认路径首次中断处也仅多覆盖 2/0/4 个候选，验证批次仍有空位的中断处净增均为零。候选保留在离线实验中。采样峰值 40.72 GiB，系统可用统一内存最低 52%，采样和测试均无新增换出或交换。见 `runs/mtp-cross-request-residual.log`。

从无损描述符的首末地址识别整个 K 分片能否顺序读取。保留原有 BF16 指数和 FP32 累加，四层真实 gate/up 和 down 的 6/8/11/16 位置输出均逐位一致。8 位置 down 微基准从 0.283 降至 0.272 ms，gate/up 则从 0.478 增至最快候选的 0.588 ms。整模只测试 6/8 位置的 down 候选，三种验证形状的隐藏状态、候选、词表提示和提交缓存均逐位一致。首次整段对照因测试模型复制时更换了草拟词表类型而失效，修正后的代码和英文 4096/1024 为 32.01 → 32.13、20.23 → 20.43 token/s，输出和 241/271 轮复现。两组均为 fair，解码峰值 41.36 GiB，可用统一内存最低 52%，无新增换出或交换。候选保留在基准中。见 `runs/bf16-clean-slice-projection.log` 和 `runs/ffn-clean-narrow-ab-retest.log`。

同一代码 4K 前缀的两次测量均报告 fair。8 位置基线为 72.51/81.10 ms，16 位置为 107.74/137.31 ms，扩宽代价分别为 35.23/56.21 ms。在存量中文、代码和英文固定轨迹中，事后知道正确续写时，分别可将 25/29/8 个 16 位置验证轮缩到 8 而保持当轮确认数。用两档实测成本折算，目标阶段约省 0.88–1.41、1.02–1.63、0.28–0.45 s。该回放使用存量轨迹与事后信息，单轮缩窄也可能影响后续调度，数值用于衡量宽批决策的潜在收益。见 `runs/frontier-hot-width-budget.log`。

用同一份原版 BF16 模型交替验证动态宽批门槛。每轮只读取已完成轮次的窄批/宽批验证耗时中位数和每个确认 token 的整轮耗时，从而把扩宽的预计成本换算成新增确认数，门槛限于 0.72–1.30。代码与英文 4096/1024 的宽批从 71/41 降至 45/19，总验证从 241/271 增至 247/278，完整解码分别省 0.55/0.47 s。中文宽批由 70 降至 50，总验证从 213 增至 223，解码多用 0.31 s。三组输出逐 token 一致。代码过程中由 nominal 转为 fair，英文和中文均为 fair。候选保留在试验中，解码峰值 41.36 GiB、可用统一内存最低 52%，无新增换出或交换。见 `runs/frontier-timed-width-ab.log`。

用历史已记录的扩宽决策模拟两档实测验证成本。动态门槛在代码固定轨迹上估算省 0.61–1.27 s，中文估算为增加 0.11 s 到减少 0.18 s。再用此前宽批实际多确认的 token 数调整门槛，代码估算省 0.38–0.95 s，中文两档仍各多用约 0.11–0.12 s。该估算使用固定轮边界和限定成本，不计调度改变对后续轮次的影响。见 `runs/frontier-timed-width-replay.log` 和 `runs/frontier-timed-width-shadow-replay.log`。

在单份原版 BF16 模型上实现了 24/32 位置验证原型。每层让注意力与 DeltaNet 处理整批，投影沿用现有无损权重包及 16-token 算子分块。在代码 4K 前缀与已有 512 个续写 token 的两处状态，24 位置线性为 157.92–158.13 ms，32 位置线性为 174.53–175.97 ms，较各自两次串行目标验证减少约 10–15 ms。24/32 位置双分叉整轮分别为 156.48–157.75 ms、174.23–175.43 ms。线性隐藏状态与完整词表预测逐位一致。分叉提交后检查了实际可读的 K/V、卷积状态、DeltaNet 状态、下一轮隐藏状态和预测，立即提交及延迟提交均逐位一致。首轮审计曾把已丢弃的预留 K/V 槽计入差异，复核按有效长度检查并执行下一轮。两次试验热状态 nominal，峰值 41.21 GiB，可用统一内存最低 52%，无新增换出或交换。见 `runs/verify-bf16-wide-panels.log` 和 `runs/verify-bf16-wide-panels-commit.log`。

按本机 32 位置双分叉成本，草稿和提交完全免费的情景下，100 token/s 仍需平均确认 17.4–17.5 个 token。若仅加现有同量级 BF16 草稿的约 7.95 ms 权重读取估算，每轮需确认约 18.2–18.3 个。公开的 b32 草稿报告 HumanEval 平均 15.5 token/轮，使用的是 M5 与 4 位目标；本机尚无该草稿的接收长度或速度实测。SpecPilot 的两阶段路径使用低精度预检并由 BF16 目标最终确认，本实验按原版 BF16 约束测试整批精确验证。[b32 草稿模型卡](https://huggingface.co/JonasLoos/Qwen3.8-27B-DFlash2-b32) · [SpecPilot](https://arxiv.org/html/2611.02501v1#S4.SS3)

下载固定版本 `fc65843ba218eeaa8e0dbcc922034184acadd100` 的公开 b32 草稿，校验 3,848,817,896 字节、SHA-256 `01e8c919665f563339ff41ac788497f483da5418c860c051b5fee17b801b7423` 及全部 81 个 BF16 张量。用同一代码 4096/1024 提示保留原版 BF16 主模型，从每个已确认前缀运行训练宽度 32 的草拟器，再对照 16/32 节点上限和最多两条分叉的精确验证。两组交替运行，输出均逐 token 复现原生 MTP 的已保存结果。16/32 节点分别为 32.16/19.46 token/s、258/252 轮、平均确认 3.97/4.06 个 token。草拟约 29.09/29.32 ms，主模型验证约 91.45/175.77 ms，提交约 2.73/3.52 ms。按整轮实测到 100 token/s 每轮分别需确认 12.33/20.86 个。允许在实际轮首直接挑出每深度 top-16 中的正确 token、忽略分叉占位，固定轮首平均也只有 6.28/6.33 个。两组 nominal，加载峰值 45.09 GiB，解码峰值 43.93 GiB，可用统一内存最低 49%。解码期间换出和交换计数不变，完整实验另有 102 次 pageout 增量，无 swap 增量。原版 BF16 MTP 服务恢复并通过健康检查。见 `runs/dflash2-b32-bf16-code-ab.log`。[b32 草稿发布页](https://huggingface.co/JonasLoos/Qwen3.8-27B-DFlash2-b32)

在 4096-token 固定提示词缓存上，让同一 K/V 头的两个连续查询头和两个候选位置共享一次读取。五种 6–8 位置的线性、分叉及预留后缀形状均与正式内核逐位一致。名义热状态下，6 位置线性注意力由 0.392 增至 0.707 ms，8 位置双分叉由 0.453 增至 0.889 ms。候选留在微基准，未用正式模型测吞吐。测试用 8 GiB 上限且未加载模型，系统可用内存最低 74%，正式 BF16 服务已恢复并通过健康检查。见 `runs/gqa-four-queries.log`。

在保存的三组 4096/1024 BF16 轨迹上回放两种短语草拟。第一种从已知历史查找跨 tokenizer 边界的 16–32 字节后缀并重新分词，第二种允许历史短语有一个不同的数字或 ASCII 标识符，并将捕获值替换为当前值。每轮只读取提示和此前确认的内容。保留原验证档位，在同档剩余空间加入候选，或在满批时替换一个低分叶子。中文、代码、英文均无额外确认 token。对 202/236/273 个首漏，若预知哪个记录正确，两种候选合计只在代码补中 1 个 token、英文补中 5 个。该上界限于测试的检索索引和最近 12 个匹配，固定轨迹不计后续轮边界变化。CPU 进程峰值约 268 MiB，未加载模型或占用 Metal。见 `runs/frontier-byte-template-replay.log`。

同一批 711 个首漏中，383 个的正确 token 已在该位置记录的前八名，195 个在前八名之外，133 个位置未打分。这是现有固定树的分类。即使补中首漏，是否可减少在线验证轮数仍取决于同批剩余位置和后续草拟。见 `runs/frontier-byte-template-replay.log`。

用动态规划在原节点数和最多两条附加分支的约束下重选整棵验证树。已展开状态直接复用，未展开的已打分候选只作为叶子，保留历史短语轮。按负载互相留出，用其他两组的 BF16 验证标签校准 MTP 前四名。固定轨迹上英文/中文/代码分别净增 13/0/0 个确认 token；前八名用同一目标函数未增加覆盖收益。用一份模型交替实跑英文和代码 4096/1024，候选和基线输出逐 token 一致。英文 41.87 → 41.62 token/s，代码 45.58 → 45.40 token/s，两组均多了一轮实际主模型验证。重排节点的 CPU 开销分别为 34 和 23 ms。实验留在夹具中，正式服务使用原调度并通过健康检查。对照期间热状态 nominal，MLX 峰值 41.36 GiB，系统可用内存最低 52%，无新增换出或交换。见 `runs/frontier-tree-budget-dp.log` 和 `runs/frontier-tree-budget-dp-ab.log`。

在三组原版 BF16 4096/1024 轨迹上检验 MTP 评分空间与递推输入空间的差异。保留原 MTP 词表打分，只在下一步递推前将隐藏状态乘主模型末尾 RMSNorm 通道增益的倒数平方根或倒数。三组基线逐 token 匹配记录并复现中文/代码/英文 213/241/271 轮。以已确认 token 回放 86 个最多十二步的窗口，中文/代码/英文连续前四名覆盖分别由 6.10/6.05/5.23 降至 4.58/4.20/4.41（倒数平方根），进一步降至 3.45/3.42/3.72（倒数）。回放用于衡量草拟命中，候选未接入在线调度。采集峰值 41.50 GiB，回放峰值 39.86 GiB，可用统一内存最低 51%，三个回放均无新增换出或交换。见 `runs/mtp-target-norm-transport.log`。

从原生 MTP 的实际递推采集每组 938 个状态。以当前草稿状态和每个窗口起点的已确认隐藏状态为输入，拟合 64 维岭回归残差，并对另一版本加上原版词表权重下的正确 token 与最强错误候选的对比分数损失。每次仅用其他两组拟合 1876 行，留出整类负载做闭环递推回放，原生 BF16 词表打分方式保持一致。中文/代码/英文原生的连续前四名均值为 6.105/6.047/5.233；同组候选中的最好成绩分别为 5.988/6.000/5.093。训练损失下降未转化为留出负载的连续命中收益，候选未上线。三组基线输出逐 token 匹配，分别复现 213/241/271 轮。采集峰值 41.85 GiB，拟合与回放峰值 41.04 GiB，可用统一内存最低 52%，回放期间无新增换出或交换，服务恢复后通过健康检查。见 `runs/mtp-rollout-adapter.log`。

保持 FFN down 的 34 个 K 分片和 FP32 归约顺序，直接以 SIMD 片段布局向量写回，再做原来的两次 BF16 舍入与残差相加。比较了线程组转置和五种分片交错。四层真实权重的 6/8/11/16 位置均逐位通过；最快布局在 8 位置由 255.10 降至 254.97 µs，在 6/11/16 位置分别由 253.33/320.44/334.43 增至 255.35/324.65/334.81 µs。候选仅用于微基准。首次用了 Metal 保留字作局部变量名，修正后跑完全部形状。测试热状态 fair，MLX 峰值 1.91 GiB，可用统一内存最低 71%。见 `runs/down-fragment-partial-layout.log`。

在代码首轮和英文输出约过半的两组实际 8 节点树形批次上，分别逐位重放全部 64 层。对 FFN down 按激活范数或每八输出行的权重范数挑最小权重包，比较原 down 输出以及最终 BF16 残差。只省最弱一包时，加入残差后仍一致的元素占比由约 8.4%–9.5%（down 输出）增到 26.1%–30.7%，但 81,920 个八输出行块中没有一块在所有候选上逐位一致。每块试省 8 或 32 包也无整块匹配。使用分包 L2 范数乘积加浮点余量的区间筛查没有确认任何整块。该筛查只用于判断改造空间，候选未上线。审计峰值 40.77 GiB，计时区间内可用统一内存最低 53%，无新增换出或交换。见 `runs/ffn-residual-skip-audit.log`。

比较让主模型每 8、16、32 层提前提交短批验证。代码 4K 前缀和已推进 512 个输出的两处状态上，6 位置线性与 8/16 位置分叉的隐藏状态、全词表预测、词表提示、下一轮隐藏状态和提交缓存均逐位一致，覆盖立即及延迟 K/V 提交。每 8 层提交时，两处 8 位置验证分别由 72.24/74.38 降至 70.45/71.41 ms。单份模型交替生成完整代码 4096/1024 请求，解码由 44.10 提升至 46.09 token/s，主模型验证累计由 19.51 降至 18.56 s，1024 个输出 token、241 轮验证及草拟统计均一致。对照峰值 41.01 GiB，可用统一内存最低 52%，无新增换出或交换。见 `runs/verify-async-segments.log`。

优化配置已默认启用每 8 层异步提交，整轮末尾继续等待完整验证结果后再提交 K/V 与 DeltaNet 状态。正式 HTTP 中文、代码、英文 4096/1024 请求的解码分别为 50.22/42.06/31.28 token/s，首 token 为 13.30/14.00/16.28 s。三个请求均为 fair 热状态，输出哈希、213/241/271 轮及草拟和 K/V 延迟提交计数均复现。请求峰值 41.50 GiB，可用统一内存最低 52%，无新增换出或交换。短请求回退、非法调度、超上下文、Metal 互斥和健康检查均通过。见 `runs/http-verify-async-segments.log`。

用一份模型按验证轮交替复测同步与已上线的每 8 层异步提交。两个负载均输入 4096、输出 1024 token，输出逐 token 和草拟统计一致。英文解码 17.59 → 17.77 token/s，累计主模型验证 51.85 → 51.29 s，均为 271 轮。中文解码 17.65 → 17.74 token/s，验证 52.27 → 51.94 s，均为 213 轮。该次持续负载全程为 fair，绝对速度低于先前 HTTP 样本。另在不同前缀长度的 8/16 位置树、立即及延迟 K/V 提交上逐位检查每 4 层提交，未显示稳定的计时收益。正式配置保留 8 层。两组交替生成峰值最高 41.36 GiB，可用统一内存最低 52%，无新增换出或交换。见 `runs/verify-async-cross-workloads.log`。

将 38.94 GB 主模型权重只读扫描和固定 8 节点验证插入同一代码 4096/1024 请求的阶段边界。基线逐 token 复现、241 轮，扣除探针时间的解码为 45.74 token/s。解码前、解码后、空闲 40 秒三次只读扫描分别为 53.54/54.85/55.60 ms，固定树验证为 70.59/79.09/70.91 ms。固定验算恢复时，权重只读扫描未恢复到起点水平；这个对照只说明本次验证时延波动不能由扫描时间解释。四层实际 FFN gate/up 与 down 的原生 BF16 操作数版本在 8/16 位置均逐位匹配。微基准收益较小且受采样阶段影响，未接入正式模型。对照阶段峰值 40.76 GiB，可用内存最低 51%，解码及探针无新增换出或交换；模型准备阶段新增 300 次 pageout，交换计数不变。首次用元组传入树形父索引导致初始化停止，改为现行 API 所用的列表后完成。服务恢复并通过健康检查。见 `runs/verify-drift-read-compute.log`。

将短批 DeltaNet 的 RMSNorm、SiLU 门控和投影片段重排合为一个 Metal kernel，复现归一化过程的两次 BF16 舍入与最终门控写回。2/6/8/11/16 位置通过逐位校验；8 位置微基准由 64.32 降至 57.06 µs。在代码 4K 前缀和续写到 512 token 的两处状态，6 位置线性、8/16 位置分叉的隐藏状态、预测、词表提示、提交缓存和下一轮结果逐位一致，覆盖立即和延迟提交。8 位置两处整模配对分别省 0.124、0.136 ms，约 0.18%–0.19%，实验止于固定批次。单份模型的峰值 45.09 GiB，整模采样可用统一内存最低 52%，全程 nominal，换出和交换计数不变。见 `runs/verify-delta-gate-fragments.log`。[MLX 0.32.4 的 RMSNorm 内核](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/kernels/rms_norm.metal)

把 9–16 位置的两个 FFN 激活面板按 SIMD lane 相邻存放。残差归一化直接写入新布局，gate/up 的 SwiGLU 写回沿用它，让 down 同样一次读入两面板。四层真实权重上的 9/11/13/16 位置，残差、BF16 SwiGLU 和 FP32 分片逐位一致。16 位置微基准由 0.804 降至 0.786 ms，原生 BF16 操作数的变体为 0.797 ms。代码 4K 前缀和已有 512 个续写 token 的两处状态中，整模 11/16 位置每轮省 1.06–1.13 ms，立即及延迟提交逐位一致。完整 4096/1024 交替对照为 47.67 → 47.36 token/s，累计目标验证 18.24 → 18.40 s，均复现 1024 个输出 token 和 241 轮。保留原布局。微基准峰值 4.82 GiB，完整解码峰值 41.01 GiB、系统可用内存最低 52%，无新增换出或交换。服务恢复后通过健康检查。见 `runs/ffn-interleaved-activation-panels.log`。

复测交错激活时，只在 9–16 位置进入候选 FFN，8 位置及以下直接复用同一份正式模型和计算图。完整代码 4096/1024 交替对照按实际批宽计时，71 个宽批的配对中位数每轮省 1.04 ms，累计省 67.86 ms。170 个窄批的配对中位数相差约 0.024 ms，累计验证抵消约 32.23 ms；全段验证仍节省 35.63 ms。验证以外耗时多约 86.67 ms，解码为 46.77 → 46.66 token/s。两路逐 token 相同，均验证 241 轮，草拟计数一致。正式仍保留原布局。解码峰值 41.01 GiB，可用统一内存最低 52%，热状态 nominal，无新增换出或交换。见 `runs/ffn-panels-live-width-isolated.log`。

用 `scripts/bench_ready.py` 接入 ReaDy 的 BF16 草稿和 DeltaNet 状态读取器，固定草稿权重版本 `537abff89429af2254ab4fbf968ebce0fbd946ff`，校验全部 67 个张量的精度及权重文件的 SHA-256。实验使用固定长度和逐步贪心 Markov 草拟。原版 BF16 主模型对同一代码 4096/1024 请求验证 8/16 个位置时，分别得到 33.14/28.68 token/s、319/296 轮、平均确认 3.21/3.46 个 token，均逐 token 复现原生 MTP。草拟、主模型验证、提交每轮分别约为 20.75/71.42/4.57 ms 和 29.37/87.43/3.68 ms。同一提示此前的 MTP 基线为 47.67 token/s。单份主模型的装载和预填充峰值 45.09 GiB，解码峰值最高 43.30 GiB，系统可用统一内存最低 50%，两组解码均为 nominal，分别新增 1/0 次 pageout，交换计数无变化。正式 BF16 MTP 服务已恢复并通过健康检查。见 `runs/ready-bf16-local.log`。[ReaDy 实现](https://github.com/StartLuxLabs/ReaDy)

用 `scripts/bench_dspine.py` 将腾讯 DSpine-b16 接入同一原版 BF16 验证器，草稿固定版本 `ea909c0f5fab8865f9e14600334f55f7ce47308f`，校验 4,051,831,880 字节、67 个 BF16 权重张量和 4 个整数版本标记。相邻位置的 BF16 特征写入通过 CPU 逐位校验。草拟采用发布方 MLX 的贪心分解打分，只计算每位置入围的 16 个词表项所需的精修分数。代码 4096/1024 的线性 8、线性 16、最多两处分叉的 16 位置验证分别为 39.16/33.59/34.91 token/s、273/248/234 轮、平均确认 3.75/4.13/4.37 个 token，均逐 token 复现原生 MTP。16 位置树有 36 轮通过旁支确认续写，仍慢于 8 位置线性。原生 MTP 在同一提示的先前基线为 47.67 token/s。装载及预填充峰值为 45.09 GiB，解码峰值最高 43.61 GiB，系统可用统一内存最低 50%，三组均为 nominal，分别新增 23/14/14 次 pageout，交换计数无变化。正式 BF16 MTP 服务已恢复并通过健康检查。见 `runs/dspine-bf16-code.log`。[DSpine 权重](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/tree/ea909c0f5fab8865f9e14600334f55f7ce47308f) · [MLX 参考实现](https://github.com/Tencent/DSpine/tree/be16c4fdb10aa23b4877d4b212aec09895840a06/dspine_mlx)

DeltaNet 可在原 FP32 状态槽中保存任意一个验证节点。当它位于已确认路径上时，仅重放其后的状态更新。七组线性和分叉合成输入逐位匹配 BF16 输出、缓存提交和下一轮结果。在保存的中文、代码、英文 4096/1024 验证路径上，将草拟节点编号映射到实际验证顺序并校验每条因果路径。用 48 层真实尺寸的合成激活隔离回放 213/243/275 轮，按草拟概率选择状态后，需要重放的轮数从 186/218/249 降为 155/200/213。递推加提交累计节省 21.91/15.59/29.14 ms，逐轮检查的 BF16 输出和 FP32 提交状态均逐位一致。完整请求仍使用原状态保存策略。实验未装载主模型，峰值 0.47 GiB，系统可用统一内存最低 73%，全程 nominal，无新增换出或交换。BF16 MTP 服务恢复并通过健康检查。见 `runs/delta-commit-checkpoint-micro.log` 和 `runs/delta-commit-checkpoint-timed-replay.log`。

用 `scripts/audit_dspine_complement.py` 重放代码 4096/1024 的 243 个 MTP 验证批次。原版 BF16 主模型逐 token 复现全部输出，再用 DSpine 原版 BF16 草稿评估 1023 个真实前缀的条件排名。243 个 MTP 轮首中有 186 个能从 DSpine 前 16 名延伸已确认路径，其中 85 个的首个延伸 token 排第一，21 个同时可在现有验证档位和分叉上限内补入。按实测分项成本，预知后续并允许重新选择轮边界时，在有记录的 MTP 起点与各起点的 DSpine 树之间切换估算为 52.22 token/s。保留现有树、用空位接入 DSpine 首选或前 16 名的估算分别为 52.88 和 59.85 token/s。忽略旁支占位、直接从前 16 名挑出正确候选的估算为 73.00 token/s。最后一个情景若令草拟和提交耗时为零，仍需 10.09 s 的验证时间，距 100 token/s 只余约 0.14 s。按 MTP 自身的路径结束概率选择接入点时，六档阈值在固定轨迹上的估计收益均为负，可见互补的候选仍需要更准确的选择方法。这些估算仅覆盖保存的候选和采样请求。峰值 45.09 GiB，可用统一内存最低 51%，无新增换出或交换。正式 BF16 MTP 服务已恢复并通过健康检查。见 `runs/dspine-mtp-complement.log`、`runs/dspine-mtp-complement.json` 和 `runs/dspine-mtp-graft-gate.log`。

DSpine 的 `--packed-ffn` 复用主模型的无损 BF16 Metal 前馈内核。五层 gate/up 和 down 的权重全部逐位回读，载荷由 2.674 GB 减至 2.037 GB。合成激活上的五层合计耗时在 8/16 位置分别为 9.22→3.26 ms、8.97→3.96 ms。内核的累加顺序使部分草稿激活和候选发生变化。用一份原版 BF16 主模型按轮交替运行 16 位置树，代码 4096/1024 的草拟为 26.46→20.85 ms/轮，解码为 35.84→37.78 token/s。两路均验证 234 轮、逐 token 复现原生 MTP。只保留压缩 FFN 的 8 位置线性独立运行达到 41.55 token/s，草拟为 13.95 ms/轮，273 轮及全部输出复现。16 位置混合调用在原触发规则下最多容许 15.96 ms/次。把保存的候选截到前七步，六档规则中最高只能容许 3.55 ms/次，它们都给额外验证成本按零计。独立 8 位置装载峰值 48.27 GiB，解码峰值 42.97 GiB，可用统一内存最低 50%。交替对照准备峰值 50.76 GiB，可用内存最低 49%，准备阶段有 36 次新增 pageout。两个运行的解码阶段均无新增换出或交换，正式 BF16 MTP 服务恢复并通过健康检查。见 `runs/dspine-lossless-ffn-ab.log`、`runs/dspine-lossless-ffn-chain8.log` 和 `runs/dspine-packed-graft-break-even.log`。

DSpine 的 `--packed-projections` 将同输入的 Q/K/V 合批投影，K/V 更新复用同一 BF16 载荷的描述符切片，短批上下文的 `fc` 及注意力输出也走无损权重包。五层权重和共享 K/V 视图均通过逐位回读，32 位置上下文回退输出逐位匹配。16 位置五层注意力投影由 2.78 降至 1.11 ms，4 位置上下文由 1.07 降至 0.80 ms。代码 4096/1024 的 16 位置树同场对照，草拟从 20.57 降到 19.50 ms/轮，解码从 37.87 升到 38.62 token/s，验证从 234 降到 233 轮。累计提交耗时从 1.40 增到 1.47 s，两组输出均逐 token 复现原生 MTP。见 `runs/dspine-packed-projections-ab.log`。

启用上述无损投影后，`--short-draft-vocab` 复用 32,768 行 BF16 草拟权重视图，并补入 118 行提示词中的 token，额外行占约 1.21 MB。同场用完整词表和短词表交替草拟，主模型均用完整 BF16 词表确认输出。4K 固定前缀的一次草拟为 17.61→14.88 ms，完整代码 4096/1024 的平均草拟为 19.78→16.94 ms/轮，解码为 38.00→39.49 token/s，均为 233 轮并逐 token 复现原生 MTP。保存轨迹上按六档 MTP 概率触发时，允许使用原先补中的候选、扣除必要扩宽后，最高盈亏平衡点仍为 15.96 ms/次。两轮对照只加载一份主模型，准备峰值最高 48.83 GiB，解码峰值最高 44.07 GiB，可用统一内存最低 50%。两轮准备期分别新增 53/6 次 pageout，解码期间换出和交换计数不变。正式 BF16 MTP 服务恢复并通过健康检查。见 `runs/dspine-short-vocab-ab.log` 和 `runs/dspine-short-vocab-graft-cost.log`。

DSpine `--wide-cost-ratio` 从同一棵按累计置信度构造的树取前 8 个或全部 16 个验证节点，用两档节点概率质量之比决定是否扩宽。两档都用完整的 16 位置草拟块。代码 4096/1024 的 1023 个真实前缀回放中，按两档名义成本比 1.17 扩宽会选中 953 个前缀，其中 648 个未多确认 token。只在开始一轮时预知两档确认数、仍按完整接受长度推进，最快估算为 45.13 token/s；固定 8 和直接用成本比的估算为 40.19 和 39.33 token/s。完整请求按轮交替实测为 40.14 和 39.27 token/s，分别验证 270 和 235 轮，后一组有 222 次宽批，两组均逐 token 复现原生 MTP。装载峰值 48.44 GiB，解码峰值 43.48 GiB，可用统一内存最低 50%，解码无新增换出或交换。正式 BF16 MTP 服务恢复。见 `runs/dspine-adaptive-width-ab.log` 和 `runs/dspine-adaptive-width-replay.json`。

同一代码轨迹分四段留出，从其余三段选择门槛，四段估算累计省 1.28 s。使用四段选出门槛的中位数 1.70，与固定 8 位置按轮交替复测完整 4096/1024 请求，解码由 37.86 升至 39.85 token/s，验证由 270 降至 249 轮，其中 31 轮用 16 个位置。全部输出逐 token 复现原生 MTP。该门槛在同一代码轨迹校准，收益仅在这一请求完成实测。两组全程 fair，共享一份 BF16 主模型，解码峰值 43.68 GiB，可用统一内存最低 50%，解码无新增换出或交换，准备期有 61 次新增 pageout，交换计数无变化。正式 BF16 MTP 服务恢复并通过健康检查。见 `runs/dspine-adaptive-width-thresholds.log` 和 `runs/dspine-adaptive-width-calibrated.log`。

`scripts/bench_dspine_reuse.py` 用主模型刚确认的路径重新对齐 DSpine 的剩余候选，允许跳过一次草稿前向，继续由原版 BF16 主模型验证。1023 个真实前缀的回放中，按锚点条件排名前 1/4/16 复用的估算分别为 41.81/40.14/40.07 token/s，每轮重算估算为 42.72 token/s。回放沿用测得的分项成本，每次复用另计 1 ms 排序；允许预知确认长度时，使用同一批候选和成本的估算为 45.92 token/s。完整代码 4096/1024 交替对照中，排名第一才复用的策略两次实测分别为 42.37→42.92 和 42.19→42.78 token/s。每次跳过 14 次草稿前向，主模型由 249 增至 254 轮验证，复测的 14 次复用共确认 38 个 token。四组均逐 token 复现原生 MTP。试验共享一份 BF16 主模型，装载及预填充峰值 48.44 GiB，解码峰值最高 43.72 GiB，可用统一内存最低 50%。两次准备期分别新增 3/10 次 pageout，解码期间无新增换出，交换计数全程无变化。正式 BF16 MTP 服务恢复并通过健康检查。见 `runs/dspine-lattice-reuse-ab.log`、`runs/dspine-lattice-reuse-retest.log` 和 `runs/dspine-lattice-reuse-ceiling.log`。

`scripts/bench_dspine_context.py` 缩短 DSpine 草拟注意力读取的 K/V，保持原始 RoPE 位置及主模型的完整 BF16 验证。对代码 4096/1024 的 1023 个真实前缀比较三种窗口，由前半段回放选出最早 128 加最近 2048 个位置。草拟中位数由 15.28 降至 11.92 ms，完整回放预计验证由 247 增至 258 轮。实际交替生成代码由 41.80 升至 42.75 token/s，249→257 轮，草拟累计 4.34→3.47 s。直接在英文 4096/1024 复测同一窗口，解码由 35.39 升至 37.64 token/s，272→269 轮，草拟累计 5.63→4.19 s。四组均复现原生 MTP 的 1024 个输出 token，热状态为 fair。每次只加载一份主模型，准备峰值 48.44 GiB，解码峰值最高 43.68 GiB，可用统一内存最低 50%。准备阶段分别新增 53/11 次 pageout，解码无新增换出，交换计数全程无变化。正式 BF16 MTP 服务已恢复。见 `runs/dspine-context-window-code-ab.log` 和 `runs/dspine-context-window-english-ab.log`。

`scripts/bench_mtp_context.py` 在原生 MTP 的分段注意力中比较完整提示词 K/V 与最早 128 加末尾 2048 个提示词位置。一次准备紧凑的草拟视图，持续保留全部续写 K/V 和原来的绝对位置。主模型从同一份 4K 预填充状态做完整 BF16 验证。代码、英文 4096/1024 的解码分别为 37.79→37.48、23.20→22.87 token/s，主模型验证分别由 241→244、271→279 轮。四组输出均逐 token 复现原版，热状态 fair。用 1/512/1024 个续写位置隔离测量时，草拟注意力每次只省约 13–14 µs，按两组原草拟打分量折算仅约 0.02–0.03 s。短视图保留在实验路径。整模装载峰值 45.09 GiB，解码峰值 41.23 GiB，可用统一内存最低 52%，整模实验无新增换出或交换。注意力微基准未加载权重，峰值约 0.067 GiB。见 `runs/mtp-context-window-ab.log` 和 `runs/mtp-context-attention-micro.log`。

`scripts/bench_verify_gpu_timing.py` 和 `scripts/metal_verify_timing.m` 在独立基准进程记录 Metal 命令缓冲提交与完成时间戳。代码、英文 4096/1024 解码复现 241/271 轮和原版输出。采样的 44 个实际验证轮中，命令缓冲 GPU 区间并集占单轮墙钟时间的中位数为 99.34%，缓冲之间空档中位数 0.45 ms。生成前后重放原 4K 缓存、相同 token 和树路由，8 位置墙钟时间中位数分别从 74.07/77.66 升至 118.14/139.69 ms，16 位置从 104.27/112.65 升至 171.12/212.82 ms。各对照的预测哈希一致，系统均报告 fair。另做 97 次固定 8 位置验证，在没有 MTP 草拟的情况下持续 14.11 秒。压力前、压力后和空闲 20 秒后，固定验证分别为 81.09/143.93/70.15 ms，38.94 GB 常驻权重只读扫描为 53.82/56.08/55.83 ms。固定验证的慢时段主要落在 GPU 命令缓冲区间内，硬件时钟和执行单元利用率仍未测量。两个实验各只加载一份模型，峰值最高 45.09 GiB，可用统一内存最低 52%，无新增换出或交换。正式 BF16 MTP 服务恢复并通过健康检查。见 `runs/verify-gpu-timing.log` 和 `runs/verify-gpu-fixed-stress.log`。

`scripts/bench_verify_buffer_limits.py` 比较 MLX 0.32.4 默认的 50 操作、50 MiB 上限与实验的 128 操作、1024 MiB。固定代码 4K 前缀的 8/16 位置验证在跨进程对照中把每轮约 223–282 个缓冲减少为 18 个，三个进程的隐藏状态和预测哈希一致。完整英文 4096/1024 按默认、实验、默认串行跑得 32.95/24.86/17.80 token/s，均为 271 轮并逐 token 复现原版。三次热状态均记为 fair，默认两次的差异大于任何可归因于该实验的变化。随后在单份模型中，用 `scripts/metal_buffer_limits.cpp` 在 GPU 同步后切换上限，对同一缓存和同一批候选逐轮配对，覆盖 fair 下持续 192 轮的前、中、后。每档宽度各 18 对，8 位置的大缓冲耗时配对中位数少 0.047 ms，16 位置多 0.992 ms，每轮缓冲约 280 个降到 18 个。所有对照隐状态及预测逐位一致。生产服务继续使用默认上限。实验装载峰值最高 45.40 GiB，完整解码峰值最高 42.02 GiB，可用统一内存最低 52%，无新增换出或交换，正式 BF16 MTP 服务恢复并通过健康检查。见 `runs/verify-command-buffer-limits-ab.log`、`runs/verify-command-buffer-limits-live.log` 和 `runs/verify-command-buffer-limits-paired.log`。[MLX 0.32.4 的提交阈值](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/device.cpp#L502-L505)

`scripts/bench_frontier_phrase_width.py` 用其他两类请求的已验证结果校准复制短语的命中率，根据还需确认的 token 数估计是否值得保留宽批。限制在普通短语追加的轮次，长锚定短语沿用原规则。代码固定轨迹里，按 1.45 倍宽批价格收窄 44 轮，29 轮确认数一致，累计少确认 44 个 token。按估计概率重排再裁剪，反而累计少确认 49 个。因此实跑保留原前八个候选。完整 4096/1024 代码配对在 nominal 为 46.41→46.93 token/s，完整预跑后 fair 为 30.97→31.75 token/s。两次代码均从 71 降至 27 个宽批，总验证由 241 增至 251 轮。fair 组累计目标验证由 29.01 降至 28.08 s。留出英文的 nominal 对照为 43.04→43.00 token/s，宽批 41→25，总验证 271→278。所有输出均逐 token 复现原版。固定价格留在实验路径。每次只加载一份原版 BF16 主模型，峰值 45.09 GiB，解码峰值最高 41.36 GiB，系统可用统一内存最低 52%，无新增换出或交换。见 `runs/frontier-phrase-width-replay.log`、`runs/frontier-phrase-width-code-ab.log`、`runs/frontier-phrase-width-code-warm-ab.log` 和 `runs/frontier-phrase-width-english-ab.log`。

短语宽批的 `--recent-price` 只用同一请求最近 36 个已完成轮次。分别收集 8、9–12 和 13–16 位置的验证中位耗时，加上近期草拟与提交耗时估算整轮代价。至少有 6 个窄批和所属档位的 3 个宽批样本时才允许收窄。固定轨迹回放三个负载的价格中位数约为 1.21，只收窄英文/中文/代码的 0/1/1 轮。英文 fair 的完整 4096/1024 对照为 27.23→27.58 token/s，宽批 41→36，总验证 271→272，价格中位数 1.405，22 个合格候选中因样本不足跳过 13 个、实际收窄 6 个。代码三路对照在一次预跑后由 nominal 转入 fair，基线、固定、动态为 43.62/43.68/43.68 token/s；动态价格中位数 1.215，收窄 3 轮，固定价格收窄 46 轮。同一代码请求连续两次完整预跑后的 fair 三路为 42.94/43.55/43.29 token/s，动态价格为 1.309，收窄 11 轮，宽批 71→61、总验证 241→243。所有输出均逐 token 复现原版。动态方案留在实验路径。每次只加载一份 BF16 主模型，装载峰值 45.09 GiB，解码峰值 41.36 GiB，可用统一内存最低 52%。一次准备或预跑区间新增 1 次 pageout，三次计时解码均无新增换出，交换计数全程无变化。正式 BF16 MTP 服务恢复并通过健康检查。见 `runs/frontier-phrase-recent-price-replay.log`、`runs/frontier-phrase-recent-price-english-ab.log`、`runs/frontier-phrase-recent-price-code-warm-ab.log` 和 `runs/frontier-phrase-recent-price-code-sustained-ab.log`。

`scripts/bench_verbatim_ffn.py` 从四层实际 FFN 解出原样 BF16 权重包并逐位回读。压缩、原样描述符路径、直接读取路径保持同一 FP32 SIMD 乘加与归约顺序，BF16 SwiGLU 和 FP32 down 部分和逐位一致。四层载荷由 1.629 GB 增到 2.150 GB。持续固定验证 213 轮后的 fair 状态，8 位置压缩/原样/直接读取为 2.90/3.28/3.21 ms，16 位置为 4.06/3.86/3.47 ms。随后 `scripts/bench_verbatim_ffn_live.py` 为 16 层保留原样包，仅在 16 位置验证走直接读取。代码 4096/1024 全程 fair，47 个 16 位置批配对中位数省 3.85 ms，累计省 154.62 ms；其余 194 轮累计差 6.22 ms。完整解码 32.93→33.12 token/s，两路逐 token 复现原版，241 轮的偏移和批宽一致。候选额外驻留 8.01 GiB，准备峰值 49.51 GiB，解码峰值 49.03 GiB，可用统一内存最低 48%。准备区间新增 2 次 pageout，解码无新增换出，交换计数全程无变化。首次微基准修正了过小占位数组的 Metal 地址空间，首次完整对照修正了计时夹具读取原始入口的时机。候选保留在实验路径，正式 BF16 服务恢复并通过健康检查。见 `runs/verify-verbatim-ffn-heat-retest.log` 和 `runs/verify-verbatim-ffn-direct16-live-retest.log`。

拆分直接读取的 gate/up 与 down。在 16 位置四层微基准，冻结 down 的输入激活，BF16 SwiGLU 和 FP32 down 分片均逐位匹配。两侧各 20 对中，原样内核每对都较快，配对中位数分别节省 0.520 和 0.362 ms，新增载荷为 1.333 和 0.669 GiB。单模型的代码 4096/1024 对照仍只对 16 位置批启用。16 层仅替换 down，额外驻留 2.68 GiB，47 个宽批的配对中位数省 1.31 ms、累计省 74.22 ms，整段为 29.95→29.86 token/s。仅替换 gate/up，额外驻留 5.33 GiB，宽批每轮中位数省 3.09 ms、累计省 139.57 ms，整段为 20.77→20.87 token/s。两次全程 fair，各自 241 轮的偏移、批宽和 1024 个输出 token 一致。两次基线的绝对速度不同，分别使用各自的同场对照。候选保留在实验路径。两组总峰值最高 46.39 GiB、解码峰值最高 46.34 GiB，可用统一内存最低 49%，整段均无新增换出或交换，正式 BF16 MTP 服务恢复并通过健康检查。见 `runs/verify-verbatim-ffn-projection-ab.log` 和 `runs/verify-verbatim-ffn-gate16-ab.log`。

`scripts/bench_dspine_early_code.py` 在 BF16 原权重上用第 2/3/4 层浅层读出投影和 1024 维码本产生固定 8 位置贪心候选，按需只更新实际用到的草稿 K/V 层。对代码 4096/1024 的 1023 个真实前缀，完整五层的草拟中位数和回放轮数为 11.91 ms、258 轮，三种浅层方案为 3.13/4.50/5.86 ms、557/505/463 轮；允许事后选择当前已有候选的估计最快为 46.87 token/s。四层码本实际 24.85→14.59 token/s、257→462 轮，按轮交替，较慢一侧单独跑完尾段。再用四层保留末端归一化、词表排序、第四层条件写入矩阵和与基线相同的 8/16 位置选择，回放为 258→361 轮。实际按已生成 token 数交替，解码 18.37→14.11 token/s，257→361 轮，草拟每轮 30.96→25.52 ms。两组各只加载一份原版 BF16 主模型，全部按完整 BF16 验证，逐 token 复现 1024 个输出，计时阶段为 fair。准备峰值 48.44 GiB，解码峰值最高 43.52 GiB，可用统一内存最低 50%，准备和回放分别新增 9/59 次 pageout，解码无新增换出，全程无新增交换。正式 BF16 MTP 服务恢复。见 `runs/dspine-early-code-code-ab.log` 和 `runs/dspine-early-head-code-ab.log`。[DSpine 项目说明](https://dspine-project.github.io/)
### BF16 DSpark

`scripts/bench_dspark.py` 复用原版 BF16 主模型验证器，接入 Red Hat DSpark 草稿的固定版本 `87ca2fdc67f316f6c1f9ebc7ef7bbb0ad299a4ce`。校验配置和 3,976,869,890 字节权重的 SHA-256，确认全部 62 个张量为 BF16。草稿使用八处主模型残差输出、2048 位置因果滑窗、逐位置 rank-256 Markov 贪心和固定 8 位置验证。代码 4096/1024 实测为 37.70 token/s，287 轮验证，平均确认 3.56 个 token，全部输出逐 token 复现原生 MTP。草拟、验证、提交平均为 19.20、71.14、4.21 ms/轮。此前同代码请求的 DFlash2-b32 分叉与原生 MTP 在各自运行中分别测到 32.16 和 47.67 token/s。

回放 1023 个已经确认的真实前缀，后验知道正确前缀并能保留条件候选前 2 名时，贪心草稿由 287 降到 235 轮。允许后验选择已记录的 MTP，或扩展 DSpark 前 4 名，并计入正确前缀上排名更高的候选占用的节点，8/16 节点方案最少 201/186 轮。最乐观的 186 轮统一按 `runs/current-verifier-roofline.log` 的 8 位置固定验证 74.77 ms 估算，草拟和提交成本按零计，得到约 74 token/s。回放结果用于筛选这一组草稿及搜索预算。MLX 装载、解码峰值分别为 45.09、43.38 GiB，解码可用统一内存最低 50%，解码无新增 pageout 或交换，加载到回放合计有 121 次 pageout。正式 BF16 MTP 服务已恢复。见 `runs/dspark-redhat-bf16-code.log` 和 [草稿模型卡](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark)。
### 32 位置验证的 FFN 线程组顺序

`scripts/bench_wide_projection_walk.py` 重排两个 16 位置面板的线程组，让读取相同权重块的组在提交网格中靠近。候选仍使用原来的无损 BF16 权重包、SIMD 累加和归约。四层真实 FFN 的 17/24/32 位置均通过 BF16 SwiGLU 与 FP32 down 部分和的逐位核对。32 位置两段 FFN 从 6.11 降到 5.70 ms，17 和 24 位置无收益。该实验未采集 DRAM 流量。微基准峰值 4.82 GiB，可用统一内存最低 71%，无新增换出或交换。见 `runs/verify-wide-projection-walk.log`。

`scripts/bench_wide_projection_live.py` 复用本机既有 32 位置验证和 DFlash2-b32 实验脚本，仅对 32 位置的 FFN 启用新顺序。单份 BF16 主模型上，4K 固定双分叉验证中位数由 168.99 降至 160.81 ms，隐藏状态、完整词表预测、立即提交和延迟提交后的下一轮逐位一致。两组代码 4096/1024 解码按轮交替，均为 252 轮，其中 249 轮使用 32 位置，且逐 token 复现原版 MTP。累计验证 43.35→41.41 秒，解码从 19.99 升至 20.79 token/s。该优化保留在实验路径。解码峰值 43.93 GiB，可用统一内存最低 50%，解码无新增换出或交换，整个实验有 23 次新增 pageout，无交换。正式 BF16 MTP 服务已恢复。见 `runs/verify-wide-projection-walk-live.log`。
### 原生 MTP 宽批的调度和中间层剪枝

`scripts/bench_wide_projection_walk.py --narrow` 在四层真实权重上把 9/11/13/16 位置拆成两组 8 位置线程组，比较三种相邻调度和独立替换 gate/up 或 down 的效果。SwiGLU 和 down 部分和逐位一致。两段同时调度在四个宽度都慢约 0.6–0.7 ms，单段变体的配对中位数也全部变慢。原路径在同一个 SIMD 组中一次解码权重后执行两个面板。微基准峰值 4.82 GiB，可用内存最低 71%，无新增换出或交换。见 `runs/verify-narrow-projection-walk.log`。

`scripts/bench_midlayer_tree.py` 在原版 BF16 模型上用代码 4K 前缀后两轮已保存的 16 位置验证，一轮接受主链、一轮接受旁支。已知正确路径的条件下，在第 16/32/48 层后缩到 8 位置，保留节点的隐藏值、预测以及提交后下一轮的隐藏值和有效缓存均逐位一致。第 32 层同步后缩批，两例验证配对中位数省 10.10/10.82 ms。三份完整 4096/1024 轨迹的宽批中，确认路径可无损缩到 8 的比例为 27/38（英文）、36/73（中文）、49/69（代码）。给每轮按完整 16→8 的半程差价且忽略选择开销，估算节约约 1.2%–2.4% 的累计主模型验证时间。剪枝保留在实验路径。全程仅一份主模型，峰值 45.09 GiB，可用内存最低 52%，无新增换出或交换。正式 BF16 MTP 服务恢复并通过健康检查。见 `runs/verify-midlayer-tree.log`。
### 中间层重排原生 MTP 候选

`scripts/audit_mtp_layer_readout.py` 固定原生 MTP 每个已打分位置的前八名，只用原主模型词表的对应 BF16 权重行对第 8/16/24/32/48/56 层隐藏状态打分。先保留前八名的原总概率，再测试与 MTP 分数的混合权重。每类 4096/1024 负载都用另外两类选出混合权重。第 32 层使英文/中文/代码首漏的正确 token 进入前四名的总次数从 123/64/111 变为 126/64/113。第 56 层变为 144/80/119，但 64 层验证器只剩末八层。这些计数衡量已打分候选的重排空间。

同树缩至 8 个节点时，用另外两类选择混合权重、深度与尾部概率门槛。未扣读出、同步和重排的乐观时间估算为英文省 26 ms、中文多耗 421 ms、代码多耗 29 ms，未进入正式调度。复测逐 token 复现三组 1024 个输出，回放已记录的 275/213/243 个验证轮。首次因英文归档缺两个单 token 步而触发最终长度断言，复测用同一主模型补跑这两步。六个读出深度合计中位耗时 0.42–0.75 ms/轮，回放峰值 40.78 GiB，统一内存可用最低 52%，无新增换出或交换。正式 BF16 MTP 服务已恢复并通过健康检查。见 `runs/mtp-layer-readout-retest.log`。
### 已确认的主模型隐藏状态检索

`scripts/audit_target_prefix_retrieval.py` 用原版 BF16 主模型逐 token 复现中、英、代码三组 4096/1024 请求。第 32/64 层的历史状态按提示词均值中心化后生成 BF16 检索向量。每个轮首只查待验证 token 相同且源续写已经可见的位置，最多取四个近邻；参考组按已知 token 后缀长度取源。两层向量共 104,837,120 字节，单轮查询中位数为 0.63–0.76 ms。固定 16 位置、事先选对四个近邻中的最好源时，双层检索在英文、中文、代码分别有 10、104、132 个起点可确认至少 8 个 token，精确后缀参考组为 8、109、135。

在已保存的 MTP 起点上，双层检索最多为代码的 6 轮多确认 15 个 token，为中文多确认 5 个，英文未增加。允许预知正确候选、在已记录的 MTP 轮、单 token 验证和检索间择优，并计入查询耗时时，英文双层方案的估计解码由 42.74 变为 43.33，中文由 53.12 变为 54.73，代码由 47.20 变为 49.82 token/s。这是固定轨迹的乐观估算。单份主模型回放峰值 40.58 GiB，可用统一内存最低 52%，无新增换出或交换。正式 BF16 MTP 服务已恢复并通过健康检查。见 `runs/target-prefix-retrieval.log`。
### 用原生 MTP 候选为 DSpine 提供前驱

`scripts/audit_dspine_seeded.py` 在代码 4096/1024 的 243 个已保存验证轮中取原生 MTP 主链的前 2 或 4 个候选做种子。DSpine 的实验路径在每层将种子 token 的码本特征写给下一位置，另测直接替换种子位置的输入 embedding。控制组仅在末端的条件打分使用同一批 MTP 种子。主模型逐轮复现已接受路径和全部 1024 个输出 token。2-token 种子在 242 个可测轮次中有 141 个完全正确，其中逐层注入让后续第一个 token 命中从 100 增至 108；直接替换 embedding 降至 85。4-token 种子的 71 个正确样本中，对应为 52、54、43。

同一 MTP 树保留全部原有节点，通过空位接入 DSpine 后缀。对 2-token 种子逐层注入，在所有可测轮次的原 8/16 位置档位上，事后加总多确认 27 个 token，控制组为 22 个。逐层方案草拟中位数 12.31 ms。只调用概率至少 0.75 且还可加节点的轮次，过滤后剩 101 次调用。即使事后按实际命中在原档位和扩宽之间择优，固定轨迹估算也只净省约 19 ms，未计额外草稿缓存维护。实验未进入线上调度。单份 BF16 主模型加一份 BF16 草稿的预填充峰值 48.44 GiB，回放峰值 42.35 GiB，可用统一内存最低 52%，整个过程新增 54 次 pageout，无交换。正式 BF16 MTP 服务已恢复。见 `runs/dspine-seeded-by-mtp-code.log`、`runs/dspine-seeded-capacity-replay.log` 和 [DSpine 论文](https://arxiv.org/abs/2609.36173)。
### Tandem DFlash2-b16 的原版 BF16 对照

`scripts/bench_dflash2_tandem.py` 复用本机既有 DFlash2 两分叉验证夹具。下载公开 Tandem 草稿固定版本 `98759a4995e4949c462f9a6197f59e015fd82530`，校验 3,848,817,896 字节、SHA-256 `843d7698e043118863847a560b0375e93db253f00352282d20ff0d6b6816356a`，81 个张量均为 BF16。配置与先前 b32 仅草拟块宽不同。发布方速度使用 DGX Spark 上的 NVFP4 主模型和 256/256 请求。本机使用原版 BF16 主模型和同一个代码 4096/1024 请求。两份草稿按各自训练块宽运行，固定同一 16 节点验证上限并按已生成 token 数交替。b32 → Tandem-b16 解码为 33.06 → 38.50 token/s，验证 258 → 233 轮，平均确认 3.97 → 4.39。草拟 29.15 → 23.41 ms/轮，验证均约 87.8 ms/轮。两组逐 token 复现原生 MTP，均为 nominal。双草稿预填充峰值 47.87 GiB，解码峰值 47.41 GiB，最低可用统一内存 48%。解码无新增换出或交换，全程新增 70 次 pageout、无交换。[草稿模型卡](https://huggingface.co/0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16)。

单份 Tandem-b16 草稿交替对照 16 → 8 节点，解码为 37.27 → 38.10 token/s，轮数 233 → 270，平均确认 4.39 → 3.79。两组逐 token 复现原生 MTP，热状态由 nominal 转为 fair。8 节点实测整轮耗时要求平均确认 9.95 个 token 才能达 100，超过节点上限；16 节点在上一次对照中要求 11.41 个，而在该组实际轮首使用后验正确候选遍历 DFlash2 的前 16 名、忽略旁支占位，也仅平均覆盖 6.99 个。单草稿解码峰值 43.82 GiB，最低可用统一内存 50%，解码无新增换出或交换，全程新增 11 次 pageout、无交换。正式 BF16 MTP 服务已恢复并通过健康检查。见 `runs/tandem-dflash2-b16-code-ab.log` 和 `runs/tandem-dflash2-b16-code-width-ab.log`。
### Tandem DFlash2-b16 的无损草稿投影

`scripts/bench_dflash2_packed.py` 在同一份 BF16 主模型和同一代码 4096/1024 快照上，交替比较原草稿、只打包 FFN、再加入合批 QKV、注意力输出、上下文与动态卷积投影，均用 8 节点验证。每个新载荷均逐位复原原版 BF16 权重，两组共用约 2.03 GB FFN 载荷，完整投影多 0.70 GB，原行优先权重仍供对照与长块回退。不同点积归约改变部分草稿分数，首块的 15 行首选均一致。草拟每轮 26.50→20.13→18.65 ms，验证均约 74.7 ms，解码 36.37→39.02→39.88 token/s，验证轮数 270→269→268。全部 1024 个输出 token 逐个复现原生 MTP。准备峰值 47.59 GiB，解码峰值 46.54 GiB，最低可用统一内存 48%。解码无新增页换出或交换，整个过程新增 1 次 pageout。

同一完整打包路径继续对照全草稿词表与 32K 加提示词出现的 118 个 token，主模型仍使用完整 BF16 词表。草稿额外词表行占 1.21 MB，首块所有共同评分的 BF16 logits 逐位一致。该次全程 fair，验证较前一组更慢。草拟 26.05→21.58 ms/轮，验证由 268 增至 271 轮，解码 28.98→29.66 token/s。两组均逐 token 复现原生 MTP，解码峰值 46.29 GiB，最低可用统一内存 50%，解码无新增换出或交换，全程新增 37 次 pageout。两次实验均无交换，正式 BF16 MTP 服务已恢复。见 `runs/tandem-dflash2-packed-code-ab.log` 和 `runs/tandem-dflash2-packed-vocab-code-ab.log`。
### 原生 MTP 与打包 Tandem 的同场对照

`scripts/bench_native_vs_tandem.py` 让线上 `frontier_prefetch` 原生 MTP 与打包 Tandem-b16（8 节点、32K 加提示词词表）共用原版 BF16 主模型、生产 4K 预填充和初始缓存。两路按生成位置交替，分别跑完 1024 个输出，均逐 token 复现原基线。代码是 26.73 对 25.11 token/s、241 对 271 轮；Tandem 验证累计少 0.50 s，草拟和提交合计多 2.98 s。随后同进程的中文为 20.55 对 13.43 token/s、213 对 353 轮，平均确认 4.80 对 2.90。两组交替期间均为 fair，中文组在持续运行后完成。解码峰值最高 47.29 GiB，预填充峰值最高 47.79 GiB，可用统一内存最低 50%。代码解码无新增换出或交换，中文解码有 65 次 pageout，全程新增 73 次 pageout、无交换。见 `runs/native-vs-packed-tandem-ab.log`。
在 `scripts/bench_dflash2_packed.py --trim-output` 中，固定 8 节点验证，让 DFlash2 最后一层先完成完整的 16 行注意力，再截取根和前七个候选位置进入因果卷积、FFN、词表评分和候选选择。第一块七行有限 logits 逐位一致，完整代码 4096/1024 中对照与缩短路径的 271 棵树及全部输出也一致。草拟每轮 32.66→31.08 ms，验证约 171 ms，解码为 18.25→18.43 token/s。此轮验证阶段明显慢于先前同模型的独立实验。单份 BF16 主模型和单份草稿的准备峰值 47.59 GiB，解码峰值 46.29 GiB，可用统一内存最低 50%，全程无新增换出或交换。正式 BF16 MTP 服务已恢复并通过健康检查。见 `runs/tandem-dflash2-trim-output-code-ab.log`。
### 验证分段计时与持续负载复测

`scripts/bench_verify_stage_drift.py` 对 8/16 位置双分叉夹具拆开 DeltaNet 与全注意力及归一化、FFN gate/up、down 与归约、完整词表与预取提示。分段与正式路径的隐藏值、预测、词表提示和旁支提交后的有效缓存逐位一致。逐层同步增加开销，另冻结真实层输入，同类算子合批后只同步一次。英文 8 位置的五组分别约为 19.41、7.84、23.92、13.62、3.14 ms，合计 67.94 ms，正常整轮 69.96 ms。16 位置的 FFN 两组合计 44.83 ms，分组总计 83.45 ms，正常整轮 87.15 ms。

代码、英文的 4096/1024 原生 MTP 分别为 47.25、42.59 token/s，逐 token 复现，分别验证 241、271 轮。英文之后再做超过 500 次固定宽度 16 的验证，系统由 nominal 变为 fair。压力前后，同一快照的常规 8 位置为 69.96→70.44 ms，16 位置为 87.15→87.20 ms，38.94 GB 常驻权重只读扫描为 54.07→54.09 ms。未复现先前的持续减速，无法用这些数据判断当时的减速来自哪组。首次采样在导入阶段停止，修正后完成两次复测。每次只加载一份 BF16 主模型，最高峰值 41.87 GiB，可用统一内存最低 51%，换页和交换计数不变。见 `runs/verify-stage-drift-code-retest.log` 与 `runs/verify-stage-drift-sustained.log`。

### DeltaNet 短批投影直接写入卷积输入

`scripts/bench_delta_sequence.py` 复用现有无损投影的点积和 FP32 归约，把新 QKV 与三行历史直接写入卷积和回滚使用的 BF16 缓冲，同时单独写出 Z，省去 QKV 切片后的拼接。48 层的投影输出逐位一致，同一 4K 快照完整验证的隐藏值、词表预测、预取提示、旁支提交后的卷积、FP32 DeltaNet 状态和有效 K/V 均逐位一致。8 位置中位数为 69.77→69.07 ms，16 位置为 87.04→86.51 ms。按每对相减取中位数，分别省 0.72 和 0.51 ms，两档各 12/12 次更快。

最初的代码 4096/1024 实验完整复现了 1024 个输出与 241 轮验证。首轮日志的整段生成计时包含夹具对照，固定快照的逐轮配对单独计时。预填充峰值 41.50 GiB，解码峰值 40.97 GiB，可用统一内存最低 52%，无新增换出或交换。见 `runs/delta-direct-sequence-code.log`。

接入主模型后，共用同一份权重和同一预填充快照，按生成进度交替测量两路。首次在所有短批宽启用时，代码解码由 47.53 降至 47.06 token/s，几个少见批宽的首次执行抵消了常见批宽的收益。限定为服务器启动时预热的 6、8、11、16 位置后，代码为 36.17→36.39 token/s，中文技术报告为 18.82→18.94 token/s。该轮全程为 fair，验证路径分别维持 241 和 213 轮，全部输出逐 token 一致。实验预算 64 GiB，预填充峰值 41.50 GiB，解码峰值最高 41.08 GiB，可用统一内存最低 52%，无新增换出或交换。见 `runs/delta-direct-sequence-native-ab.log` 与 `runs/delta-direct-sequence-warmed-width-ab.log`。

正式 BF16 `frontier_prefetch` 服务已启用这四个批宽的新路径。重启后短请求和健康检查通过。中文 HTTP 4096/1024 仍为 213 轮，返回文本与保存的基线一致。本次 HTTP 的绝对速度偏低，首 token 耗时 59.81 秒，解码 13.30 token/s，前后热状态均为 fair。峰值 41.50 GiB，可用统一内存最低 52%，新增 1 次 pageout、无新增交换。见 `runs/delta-direct-sequence-service-rollout.log`。

### GPU 计算时延的独立对照

同一服务进程复测中文 HTTP 4096/1024，首 token 25.28 秒，解码 25.45 token/s，仍为 213 轮及相同输出。每两秒抽样的预填充和解码 GPU 利用率中位数均为 98%，无新增换出、交换或 GPU 恢复。见 `runs/http-bf16-drift-host-telemetry.log`。

用一份原版 BF16 模型采集 Metal 命令缓冲的 GPU 时间。低开销读对照为一块 512 MiB 缓冲，其他对照为读取 38.94 GB 常驻权重、4096×4096 BF16 矩阵乘和固定 8 位置验证。解码前、确认约 768 个输出 token 和空闲 20 秒后，固定验证 GPU 时间为 76.45/161.91/68.76 ms，独立矩阵乘为 8.94/20.41/7.08 ms，权重只读扫描为 50.92/53.49/51.20 ms。固定验证的缓冲间空档中位数均低于 0.5 ms。生成 256 个 token 时，验证和矩阵乘已分别达到 138.06、16.82 ms，macOS 仍报告 nominal。空闲恢复时仍报告 fair。这组数据定位到 GPU 计算时延，未直接测硬件时钟或区分调频与其他设备状态。

对照生成逐 token 复现 1024 个输出及 213 轮，扣除探针后为 23.59 token/s。Metal 预算 64 GiB，装载峰值 45.65 GiB，解码峰值 41.37 GiB，可用统一内存最低 52%，无新增换出或交换。见 `runs/bf16-device-drift-controls.log`。

### DeltaNet 寄存器卷积与 SwiGLU 精确查表

将四抽头卷积并入无损 DeltaNet QKV 投影的写回，使用同一 SIMD 组交换 BF16 投影值，保留原序卷积乘加、SiLU 舍入和回滚历史。四层真实权重的 6/8 位置线性、8 位置单分叉和双分叉的 QKV、Z、卷积输入历史逐位一致。四层的单、双分叉配对中位数耗时分别增加 0.014/0.027 ms，6 位置线性增加 0.032 ms，8 位置线性节省 0.009 ms。该路径止于微基准。峰值 1.94 GiB，可用内存最低 72%。见 `runs/delta-shuffle-conv-micro.log`。

用原有 BF16 SwiGLU 的舍入顺序生成一张 128 KiB 表，按门控的 BF16 位型取 SiLU 后，与 BF16 up 结果相乘。全部 65,536 种门控位型与 11 种代表性 up 值的输出逐位一致，含正负零、次正规数、无穷和 NaN。四层真实 gate/up 的 6/8/11/16 位置通过逐位校验，查表配对中位数增加 0.004–0.053 ms。再做 25 秒独立矩阵乘后复测，矩阵乘为 7.31→7.10 ms，未复现真实生成中的减速，查表持平或稍慢。两次微基准峰值 4.45 GiB，可用内存最低 71%，均无新增换出或交换。见 `runs/ffn-exact-silu-table-micro.log` 与 `runs/ffn-exact-silu-table-stressed.log`。

用一份原版 BF16 模型交替生成代码 4096/1024，只在 48 个 DeltaNet 层的 6/8/11/16 位置 FFN 使用查表。固定 8/11/16 位置的隐藏状态、预测、词表提示和已提交缓存均逐位一致。整段解码为 47.42→47.53 token/s，主模型累计验证 18.184→18.145 秒，两组 241 棵验证树及全部输出一致。配对中位数在 8 位置多用约 0.003 ms，在 16 位置省约 0.085 ms。整段收益 0.23%，候选保留在实验中。这次全程 nominal，Metal 预算 64 GiB，装载峰值 45.09 GiB、解码峰值 41.01 GiB，可用统一内存最低 52%，无新增换出或交换。正式 BF16 服务已恢复并通过健康检查。见 `runs/ffn-exact-silu-table-code-ab.log`。

### 分词边界与 MTP 候选排序

只用 CPU 从已打分的前四名和已展开的后继构造等字节的 token 拆分、合并与相邻分词边界平移。回放使用保存的三个 4096/1024 请求，分别有 275、213、243 个已记录的英文、中文、代码验证轮。允许事后选出正确提案，只填原验证档位的空位或替换一个旁支叶子，最多能在 6、4、13 轮各多确认一个 token。在未记录的新起点只允许单 token 定步、允许提前结束已验证前缀时，存档轮数分别为 275→274、213→212、243→243。英文存档覆盖 1021 个解码 token，其他两组覆盖 1023 个。候选止于 CPU 回放。见 `runs/mtp-byte-segmentation-replay.log`。

用其他两类请求的已验证标签拟合 MTP 前四名概率的校正量。词法版本加入候选与父 token 的类别、候选字节数及 token ID。保持验证节点数和两条附加边上限，历史短语轮沿用原树。重选整棵树的预期收益至少 0.08 个确认 token 时才提交候选树。固定轨迹的英文、中文和代码分别净增 13、0、13 个确认 token，不使用词法特征的同门槛对照分别净增 10、1、10 个。固定轨迹的确认数用于筛选真实生成试验。CPU 拟合与回放峰值约 162 MiB。见 `runs/mtp-lexical-calibration-replay.log`。

用同一份原版 BF16 主模型和预填充缓存，按生成进度交替复跑代码 4096/1024。重排触发 54 轮，替换 77 个验证位置；实际两路均为 241 轮，宽批 71→76，输出 1024 个 token 逐个一致。累计验证时间 18.291→18.405 秒，重排计算约 98 ms，解码 45.92→45.77 token/s，两路均为 nominal。候选止于实验。首次运行因计时生成器在错误的函数绑定上构造而停止；修复后把接线检查前移至加载前并完成复跑。装载峰值 45.09 GiB，解码峰值 41.01 GiB，可用统一内存最低 52%，无新增换出或交换。见 `runs/mtp-lexical-calibration-code-retry.log`。

### 多草稿合批与分叉父节点容量

`scripts/audit_draft_round_fusion.py` 在代码 4096/1024 轨迹逐 token 复现全部输出，采集打包 BF16 Tandem 的 1023 个正确前缀排名，与同一请求已保存的 DSpine 排名和原生 MTP 的 243 个验证轮对齐。允许事后选择轮界、正确条件候选和草稿，并按同机验证及草拟耗时计价。每轮只选一种草拟路径估算为 165 轮、66.23 token/s；同轮可混合候选或以空位接入原 MTP 树，对两条附加边计价时为 155 轮、67.62 token/s。限制两处可分叉的父节点、允许同父多子且保留先于正确候选的节点时为 135 轮、74.73 token/s；取消分叉数限制但仍计节点时为 131 轮、76.30 token/s。直接从前 16 名挑正确候选且免计旁支，保留实测草拟成本时估算为 122 轮、84.49 token/s，草拟和提交免费时为 111.17 token/s。这些数值依赖事后信息和不同运行的分项计时。

排名采集只用一份主模型和 Tandem，DSpine 从已有轨迹读取。采集峰值 47.59 GiB，可用统一内存最低 51%，新增 29 次 pageout、无交换。`--reuse` 在 CPU 上复算，无需加载权重。见 `runs/bf16-draft-round-fusion.log`、`runs/bf16-draft-round-fusion-replay.log`。

`scripts/bench_dflash2_packed.py --cap 16 --compare-fork-sites` 在相同代码 4096/1024 上按已生成位置交替运行打包 BF16 Tandem，两组使用同一份主模型、草稿、分数堆和验证上限。从最多两条附加边改为最多两处可分叉父节点后，解码由 40.32 升至 41.86 token/s，验证由 233 降至 224 轮，平均确认由 4.39 增至 4.57。接受旁支由 46 增至 63 轮，新规则中有 127 个验证轮实际用了超过两条附加边。首块评分逐位一致，两侧完整输出均逐 token 复现 MTP。热状态由 nominal 转为 fair，解码峰值 46.37 GiB，准备峰值 47.59 GiB，可用统一内存最低 50%。解码无新增换出或交换，整个过程新增 7 次 pageout、无交换。见 `runs/tandem-dflash2-fork-sites-code-ab.log`。

`scripts/audit_draft_round_fusion.py --native-fork-room` 只用 CPU 审计现有 MTP 三条 4096/1024 轨迹。保留原树及 8/16 位置档位，只在已有至少两条附加边、仍有空位且最多使用两个分叉父节点时，给已确认路径的末节点增加一个孩子。即使事后从已打分项中选正确 token，英文、中文、代码的前四名也只补中 4、0、2 次首漏。允许使用诊断存档前八名时分别是 9、4、2 次。该计数不含替换原树节点、重新打分或改变后续轮界。见 `runs/mtp-sibling-room-audit.log`。

`scripts/bench_dflash2_packed.py --compare-adaptive-width` 用同一份原版 BF16 主模型和已无损打包的 Tandem 草稿，交替生成三路代码 4096/1024，均限制两处分叉父节点。动态方案按同一概率树中 16 与 8 位置的预期确认数之比决定是否扩宽，成本比 1.139 取自之前同机的分项计时。96 组 CPU 检查验证了在概率门槛强制取窄或取宽时，候选树与对应固定容量完全一致。固定 8、固定 16、动态分别得到 42.99、42.29、42.79 token/s，验证 265、224、226 轮，均逐 token 复现原生 MTP。动态有 25 轮取 8 位置，其余 201 轮宽于 8。该组动态速度与固定 8 接近，保留在实验路径。

三组解码均为 fair。全程仅一份 BF16 主模型和一份 BF16 草稿，准备峰值 47.59 GiB、解码峰值 46.61 GiB、可用统一内存最低 50%。解码无新增页换出或交换，整个实验新增 39 次 pageout、无交换。见 `runs/tandem-dflash2-adaptive-width-code-ab.log`。

### Tandem 固定窄批的 BF16 草稿评分

`scripts/bench_dflash2_packed.py --compare-ranked-head` 保持 Tandem 最后一层注意力的完整 16 行，只让根和前七个候选深度进入该层因果卷积、FFN 和词表评分。再把词表重排与分块前 16 名合并，在 BF16 权重存储顺序上直接读 logits，返回 token ID 和 BF16 分数，省去整张 logits 的词表顺序复制。真实 7/15 行 logits 的入选 ID、顺序和 BF16 位模式一致，覆盖包含正负零、无穷与 NaN 的构造样本。固定 7 行词表投影加候选选择的中位数为 2.985→2.934 ms，15 行为 3.625→3.548 ms。

代码 4096/1024 固定 8 位置、最多两处分叉父节点，三路交替生成。全 15 行、仅前七行、前七行加直接选取的草拟耗时为 17.45、16.39、16.22 ms/轮，解码为 41.93、42.78、42.95 token/s。全部 265 棵验证树的序列哈希相同，均逐 token 复现 MTP 的 1024 个输出。最高峰值 47.59 GiB，解码峰值 46.54 GiB，可用统一内存最低 50%，解码期间无新增页换出或交换，准备及预热过程新增 70 次 pageout、无交换。候选保留在实验脚本。见 `runs/tandem-dflash2-ranked-head-code-ab.log`。

### Tandem 只计算前缀查询

在同一代码输出的 243 个已记录 MTP 轮首，对齐原生 MTP 首漏和 Tandem、DSpine 的真实前缀条件排名。65 次原生 MTP 已打分但正确 token 落在前八名之外，两种草稿中至少一种排第一的有 15 次，至少一种进前 16 的有 39 次。42 次原生 MTP 末端未打分，对应为 26 和 33 次。排名回放使用真实前缀，只用于衡量模型间的覆盖。见 `runs/mtp-draft-complement-first-miss.log`。

`scripts/bench_tandem_tail.py` 在 16 位置 BF16 Tandem 草稿的末层保留所有 K/V，只计算前 8 个 Q 和对应注意力输出，输出给 8 位置验证器。Q 投影借用同一 BF16 权重载荷的描述符切片。第三组将同样的截取提前到倒数第二层，尾部仅保留该层输入供下一层读作 K/V。首块的七行 BF16 草拟 logits 在两种截取下均发生位差，首选与控制组分别一致 7/7、6/7 行。代码 4096/1024 三路按进度交替，控制、仅末层截取、两层截取分别为 43.34/43.76/43.57 token/s，草拟为 15.94/15.24/14.45 ms/轮，主模型验证 265/266/268 轮。三组逐 token 复现原生 MTP 的全部输出，验证树不同。

全程只装载一份原版 BF16 主模型和一份 BF16 草稿，Metal 上限 64 GiB，准备峰值 47.59 GiB、解码峰值 46.53 GiB，可用统一内存最低 50%。准备阶段新增 20 次 pageout，解码无新增换出，全程无新增交换。正式 BF16 MTP 服务已恢复并通过健康检查。见 `runs/tandem-prefix-query-code-ab.log`。

`--compare-fused-qkv` 在一个 Metal 调度内处理全部投影，按输出瓦片对 Q 跳过第二个 token 面板，同时保留整个草稿块的 K/V。9、11、16 输入位置的 Q/K/V、首块七行草拟 logits、完整请求的 266 棵验证树及全部输出逐位或逐 token 一致。同场 4096/1024 的草拟耗时为 17.464→17.457 ms/轮，解码为 40.19→40.50 token/s。约 0.19 秒的总差中，约 0.17 秒来自提交计时，草拟累计只少约 1.9 ms。融合保留在实验路径。准备峰值 47.59 GiB，解码峰值 46.29 GiB，可用内存最低 50%。全程新增 23 次 pageout，均在测量解码前，交换计数不变。见 `runs/tandem-fused-prefix-qkv-code-ab.log`。

`--compare-draft-widths` 在所有草稿层只输入 8 个位置。对照仍在所有层保留 16 个位置，末层只计算前缀查询。两路使用同一原版 BF16 Tandem 和主模型，验证上限均为 8，模型权重和训练配置原样保留。较短块在首块的七行草拟首选中有六行相同。代码 4096/1024 交替生成的草拟耗时从 23.18 降到 16.63 ms/轮，累积省 1.73 秒，验证由 266 增至 267 轮。解码从 28.22 升到 29.46 token/s，提升 4.4%，全部 1024 个输出逐 token 一致。两路全程 fair，Metal 上限 64 GiB，准备峰值 47.59 GiB、解码峰值 46.29 GiB，可用内存最低 50%。新增 1 次 pageout，发生于测量解码前，全程无新增交换。正式 MTP 服务已通过健康检查。见 `runs/tandem-short-draft-block-code-ab.log`。

`scripts/bench_native_vs_tandem.py --tandem-block 8 --full-vocab --ranked-head --memory-gib 64` 在同一份正式预填充和原版 BF16 主模型上交替运行原生 MTP 和 8 位置 Tandem，Tandem 直接按权重存储顺序选出全词表前 16 名。代码同场为 24.42/24.09 token/s、241/267 轮；Tandem 验证累计少 0.92 秒，草拟和提交多 1.50 秒。英文为 15.06/16.22 token/s、271/259 轮；Tandem 验证累计少 6.69 秒，尽管草拟和提交合计多 1.84 秒，整段仍快 7.7%。所有输出逐 token 一致，两个负载均为 fair，英文排在代码后。预填充峰值最高 47.79 GiB，解码峰值最高 47.37 GiB，可用内存最低 50%。英文解码新增 39 次 pageout，两组从装载到完成新增 118 次，全程无新增交换。见 `runs/native-vs-tandem-block8-ab.log`。

加 `--compare-compact-head` 时，草稿直接在 32,768 个共享 BF16 行和代码提示词的 118 个额外行内选出前 16 名；额外权重约 1.21 MB。七行固定激活上，已有带屏蔽的短词表、紧凑词表、存储顺序全词表耗时分别为 0.79、0.71、2.93 ms，前两路的选取 ID、顺序和 BF16 分数逐位一致。代码 4096/1024 三路按生成进度交替，MTP、Tandem 全词表、Tandem 紧凑词表分别为 17.70、17.58、17.98 token/s，验证 241、267、271 轮。Tandem 紧凑词表比全词表累计少用 1.57 秒草拟时间，主模型仍使用完整 BF16 词表，三路输出逐 token 一致。热状态由 fair 转为 nominal。预填充峰值 47.79 GiB，解码峰值 47.48 GiB，可用内存最低 50%。解码新增 87 次 pageout，全程 102 次，无新增交换。正式服务恢复并通过健康检查。见 `runs/native-vs-tandem-compact-head-code-ab.log`。

### Tandem 请求内候选置信校准

`scripts/bench_tandem_tail.py --compare-calibration` 在 8 位置草稿和 8 位置验证下，从本请求已验证的路径更新深度、置信区间、条件排名与前 16 名覆盖率的计数。预热后清空计数，积累 32 个已验证样本后才调整候选树优先级。主模型、草稿 logits、词表筛选和分叉上限沿用原路径。英文 4096/1024 中，校准读取 999 个已验证位置，其中 38 个正确 token 在草稿前 16 名之外。验证从 259 降至 257 轮，解码从 37.77 到 37.94 token/s，全部输出逐 token 一致。热状态从 nominal 转为 fair，准备峰值 47.59 GiB、解码峰值 46.29 GiB，系统可用内存最低 50%。准备及预热新增 20 次 pageout，测量解码无新增换出，全程无新增交换。见 `runs/tandem-rank-calibration-english-ab.log`。

代码 4096/1024 的验证从 267 降至 264 轮，解码从 23.96 到 24.25 token/s，1024 个输出 token 逐个一致。验证累计耗时从 36.63 降至 36.21 秒，草拟每轮约 20.1 ms。两路热状态均为 fair，准备及预热新增 16 次 pageout，解码无新增换出，全程无新增交换。两种负载的解码增幅为 0.46% 和 1.22%。校准保留在实验入口，正式服务已恢复并通过健康检查。见 `runs/tandem-rank-calibration-code-ab.log`。

### 宽批 FFN 原样权重的内存收益

`scripts/bench_verbatim_ffn_scale.py` 在 66 GiB Metal 预算下，用同一份原版 BF16 主模型和 4K 代码快照交替测试紧凑权重及 16 层额外驻留的原样 gate/up 与 down 包。额外驻留 8.01 GiB。8、11、16 位置各做 12 对比较，原样路径每轮分别多用 2.50、0.75、0.78 ms，只有 16 位置的一次配对更快。完整隐藏值、预测、词表预取提示和旁支提交后有效缓存逐位一致。该轮系统报告 fair，紧凑的 16 位置验证仍为 87.07 ms，和此前实测速率更低时的 fair 样本不同。按预设收益检查止于 16 层并用原生 MTP 完成 4096/1024，全部输出复现。

固定验证峰值 48.98 GiB，可用统一内存最低 48%，整个实验新增 33 次 pageout、无交换。首轮回退进入原生 MTP 后仍持有诊断备用包，已修复为回退前释放，并以 CPU 检查覆盖两种回退。正式 BF16 MTP 服务已通过健康检查。见 `runs/verify-verbatim-ffn-scale-code.log`。

### 用中间层稀疏打分收窄验证树

`scripts/bench_midlayer_prune.py` 从英文、中文各一条 4096/1024 BF16 轨迹抽取 52 轮已验证的 16 位置分叉批。用原生 MTP 路径分数和第 32 层对现有子节点的稀疏词表打分拟合边命中概率，按训练轨迹的实际确认长度确定门槛。达到门槛时选出含祖先的 8 个位置，后 32 层正常计算，完整 BF16 词表继续决定接受路径。两段各自使用原坐标提交缓存。强制选中的旁支验证了保留位置的隐藏、预测，以及实际立即或延迟 K/V 提交后的下一步隐藏均逐位一致。

第一次在线试验采用固定门槛，留出代码的 22 个可测批次没有一次缩窄。按训练轨迹校准后，代码试验在 22 批中收窄 2 批，两路均复现全部 1024 个输出和 241 轮验证，解码为 47.61 对 47.58 token/s，均为 nominal。读出和同步合计的中位观察窗口是 15.85 ms，其中包含等待前半段验证。候选留在实验路径。校准版装载峰值 45.09 GiB、代码配对解码峰值 41.08 GiB、可用统一内存最低 52%，无新增换出或交换，正式 BF16 MTP 服务已通过健康检查。见 `runs/verify-midlayer-causal-code-ab.log`、`runs/verify-midlayer-calibrated-code-ab.log`。

### Tandem 草稿候选的跨轮复用

`scripts/bench_tandem_lattice_reuse.py` 在固定 8 位置验证下，让同一份 BF16 Tandem 草稿保留最多一块旧分数。主模型刚确认的下一轮首 token 若在旧前 16 名中，读取对应条件分数；扩展方案也允许用原 BF16 码本重算旧候选对词表外前驱的条件分数。已有前驱的重算通过逐位校验。每次只跳过一轮草稿前向，未送入草稿的已确认特征在下次前向一并更新，实际最多缓存 16 行。门槛按既有窄批的分项计时和确认长度设为 3.164 个预期 token。

代码 4096/1024 三路按进度交替，逐 token 复现原生 MTP。每轮计算前七行的新草稿为 42.63 token/s、265 轮。每次保留十五行供后续复用时，只用原前驱表为 40.93 token/s、293 轮，复用 103 次、平均确认 2.95 个；允许表外前驱重打分为 39.87 token/s、304 轮，其中 22 次表外复用平均确认 2.18 个。重打分方案累计草拟省 1.15 s，目标验证多用 2.80 s。CPU 回放允许事后挑正确候选和切换轮界，在固定 8 位置、为排名领先的旁支预留节点的条件下，新草拟、可复用旧前驱、可重打分表外前驱分别估算为 54.48、56.93、58.07 token/s；忽略旁支占位后最乐观为 69.05 token/s。候选留在实验脚本。

复测只加载一份主模型和一份草稿，准备峰值 47.59 GiB，解码峰值 46.54 GiB，可用统一内存最低 50%。解码无新增换出或交换，全程新增 62 次 pageout、无交换。首次在草稿加载阶段遇到源码路径拼写错误，修正并把源码哈希检查提前后完成复测。正式 BF16 MTP 服务已通过健康检查。见 `runs/tandem-dflash2-lattice-reuse-code-retest.log`、`runs/tandem-dflash2-lattice-reuse-bound.log`。

### 原生 MTP 投影混合命中

`scripts/bench_mtp_masked_lane.py` 在单路及二、三路 MTP 输入投影中复用已有的逐 SIMD lane FP32 嵌入累加缓存，仅为未命中的 token 读取和归一化嵌入。另保留一份连续的隐藏权重半矩阵，占 50 MiB。保持原 BF16 嵌入、FP32 FMA 与 shuffle 顺序，五种缓存组合的投影逐位一致。24 对计时中，单路命中配对中位数省 0.067 ms，二、三路全命中省 0.141/0.134 ms，二、三路混合分别省 0.001 ms、多用 0.009 ms。

代码 4096/1024 按进度交替比较同一份原版 BF16 主模型。两路生成相同的 1024 个 token 和 241 棵验证树。候选实际走了 668 次单路命中、88 次多路全命中及 84 次多路混合。草拟累计 3.190→3.163 s，解码 47.09→47.36 token/s。这是一组同场结果，候选留在实验路径。预填充与解码峰值 41.50 和 41.05 GiB，可用统一内存最低 52%，无新增换出或交换。正式 BF16 MTP 服务已通过健康检查。见 `runs/mtp-mixed-lane-prefix-code-ab.log`。

### FFN 固定分片循环与 down 权重包预读

`scripts/bench_ffn_packet_loop.py` 读取四层原版 BF16 FFN 权重。对每个完整的 512 列分片固定循环上限，另测在 down 投影中先还原两个相邻权重包再执行原顺序的矩阵乘加。6、8、11、16 位置的 SwiGLU 和 down FP32 部分和均逐位一致。固定循环在每四层的配对中位数变化不超过 0.012 ms。相邻包预读在 11、16 位置分别多用 0.109、0.083 ms，两个宽度各 18 对均较慢。实验止于微基准，12 GiB 预算下峰值 4.82 GiB、可用统一内存最低 71%，无新增换出或交换。见 `runs/ffn-fixed-packet-loop-micro.log`。

### 中文解码的中途暂停对照

`scripts/bench_decode_pacing.py` 在同一份原版 BF16 主模型和同一个中文 4096-token 预填充快照上，依次做连续解码、中途暂停、连续解码。每组解码前均空闲 20 秒并将这段时间排除在解码计时之外。暂停组在确认 522 个位置后空闲 5.01 秒，计入 1024 输出的解码时间。三组分别为 52.74、41.60、53.18 token/s，验证树及输出逐 token 一致，均为 213 轮。三组全程 nominal，8 位置分段验证中位数均在 69.8–70.6 ms。这组对照没有复现此前的大幅 GPU 计算减速，结论适用于本次稳定状态。暂停仅在实验脚本中使用。

单份模型加载峰值 45.09 GiB、预填充峰值 41.50 GiB、解码峰值 40.81 GiB，64 GiB 预算下可用统一内存最低 52%，无新增换出或交换。正式 BF16 MTP 服务已恢复并通过健康检查。见 `runs/bf16-decode-pacing-chinese-aba.log`。

### DFlash2 新旧草稿按位置对齐

`scripts/audit_draft_round_fusion.py --anchor-overlap` 只用保存的代码 Tandem 和中文原版 DFlash2 条件排名做 CPU 回放。每轮照常草拟，并按本轮验证确认的位移对齐上一至两轮的候选。动态规划枚举可能的轮界，在每个位置后验选择排名较好的来源，计入该来源中排名更高的候选占位，最多使用两个分叉父节点。固定 16 位置时，代码新草稿、上一块中已有正确前驱的分数、两块且可重打分的最少轮数为 158、155、152，中文为 194、190、189。即使免计所有旁支占位，代码两块重打分为 137 轮，中文为 164 轮。对应 100 token/s 的单轮上限约 74.7/62.4 ms。该模型使用事后排名，不能直接推导实际混合排序或吞吐。CPU 峰值约 35 MiB。见 `runs/dflash2-anchor-overlap-bound.log`。

`scripts/bench_tandem_tail.py --compare-overlap --workload code` 对照 16 位置的新块和新旧块混合。每轮使用同一份原版 BF16 草稿和主模型。保留上一块的 CPU 候选分数，按已确认位移对齐。当两个块都有同一个前驱时，旧块占条件概率的 25%。两路首块 15 行 logits 逐位一致。代码 4096/1024 的解码为 43.49→43.07 token/s，验证从 224 增至 226 轮，草拟每轮从 15.82 增至 15.92 ms。混合组在 221 轮有重叠，共使用 2,470 个混合条件行，确认的候选均在相应新块的短名单中。两路全程 nominal，1024 个输出逐 token 复现原生 MTP。混合保留在实验入口。

Metal 预算 64 GiB，准备峰值 47.59 GiB、解码峰值 46.37 GiB，可用统一内存最低 50%。解码无新增换出，准备和预热新增 120 次 pageout，全程无新增交换。正式 BF16 MTP 服务已恢复并通过健康检查。见 `runs/tandem-aligned-overlap-code-ab.log`。

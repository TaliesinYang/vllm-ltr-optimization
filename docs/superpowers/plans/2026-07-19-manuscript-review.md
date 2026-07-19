# 手稿骨架审查记录(2026-07-19)

轮次:初审(rubric+Ryoo 双层,需修复)→ 6 项硬伤修复 → 复核(5 项中 3 RESOLVED,2 项细化)→ 再修 → **SKELETON-APPROVE**

## 初审全文
判决：**需修复**。教授二元 rubric 与 Ryoo 标准均未达到 APPROVE。

审核对象为 `main` 分支 commit `fd6ac998411549446691be81f11ba9f2d9f814cf`；限定范围工作区干净。现有 PDF 为 13 页，生成时间晚于 `main.tex`。全程只读，未修改文件，也未重新编译。

说明：这是对“当前骨架及其已生成 PDF”的二元验收。空白作者区满足“无 AI 正文”的红线，但不等于最终 rubric 已满足。

## 一、教授二元 rubric

| 验收项 | 判定 | 核对结果 |
|---|---:|---|
| IEEE 双栏 conference 模板 | PASS | `\documentclass[conference]{IEEEtran}`。 |
| BibTeX + IEEE style | PASS | 使用 `\bibliographystyle{IEEEtran}` 和 `\bibliography{refs}`。 |
| 标题 `Group XX:` 前缀 | PASS | 前缀存在；但实际组号、题名和作者块仍为占位符，提交前必须人工替换。 |
| 节序 | PASS | Abstract → Intro → Background → Related Work → Methodology → Evaluation → Discussion → Conclusion → Appendix A → References 顺序正确。 |
| 换页及精确页数 | **FAIL** | 注释预算齐全，但实际 PDF：Background 占第 3–5 页而非 1.5 页；Evaluation 仅第 8–9 页而非 3 页；Methodology 表格先于节标题，Evaluation 图也先于节标题。 |
| Abstract | **FAIL** | 有两个作者槽、无引用并提示 1–2 个数字，但当前 PDF 只有 `Abstract—`，没有两段正文和数字。 |
| Introduction | **FAIL** | 无小节、贡献列表位置正确，但列表仍被注释，正文为空。 |
| Background | **FAIL** | 四个小节和两图槽满足；Fig.1 是系统架构，Fig.2 严格判断为顺序 data-lineage flowchart，不满足“不得使用流程图”。 |
| Related Work | **FAIL** | 四个分类小节、无图正确；当前无相关工作正文和引用。 |
| Methodology | **FAIL** | 无图、两表正确；当前无复现方法正文/工具引用，而且表格浮到节标题之前。 |
| Evaluation | **FAIL** | 六小节和六图槽存在，但只有 Fig.3、Fig.5 是真实图；Fig.4/6/7/8 是占位框，不能算六张 graph。 |
| Discussion | **FAIL** | 六个月未来方向提示存在；当前正文为空。 |
| Conclusion | **FAIL** | 一段式提示存在；当前没有实际段落。 |
| Appendix A | **FAIL** | 独立第 12 页且有截图槽，但 GitHub 截图文件尚不存在。 |
| References | **FAIL** | 能生成 References，但正文没有引用，依靠 `\nocite{*}` 强制列出；`refs.bib` 有 17 个 metadata TODO。 |
| Matplotlib、坐标轴、脚本 | **FAIL** | 现有 Fig.1/2/3/5 均由 Matplotlib 脚本产生，Fig.3/5 有轴标签，脚本字体设为 10 pt；但四张评估图未生成。 |
| 图内字体及 caption ≥10 pt | **FAIL** | 图内字体配置为 10 pt；PDF bbox 实测 Fig.1 caption 高度约 7.97 pt，IEEEtran 默认 caption 未被提升到 10 pt。 |
| 无伪代码 | PASS | 未发现 algorithm/algorithmic/pseudocode。 |
| 无流程图 | **FAIL** | Fig.2 是“source → labels → sample → replay → training → checkpoint → serving”的过程流。 |
| 除 Intro 外无 bullet | PASS | 所有 `.tex` 中没有活动列表；唯一 `itemize` 只存在于 Introduction 注释模板。 |
| 图表顺序及正文引用 | **FAIL** | 编号本身连续（Fig.1–9、Table I–II），但没有正文 `\ref`/`\cite`。 |
| 去空白 | **FAIL** | PDF 多页只有节标题或单张浮动图；当前远未达到“remove as much whitespace as possible”。 |

**教授层判决：FAIL。**

## 二、Ryoo 标准

| 核查项 | 判定 | 证据与结论 |
|---|---:|---|
| Fig.3 来源追溯 | PASS | 来源确为 `baseline-2026-06-22/RESULTS-summary.txt`。16 qps：FCFS `17274.2`、LTR `6043.7`，比值 `2.858216`，图中 `2.86×` 正确；64 qps p99 TPOT 代价 `1400.29/171.19=8.18×`，图中 `8.2×` 正确。 |
| Fig.5 来源及图中数字 | PASS | 图脚本读取两个 matrix summary。Tier-2 三 seed 均值为 prompt-only `.587`、prompt+schema `.630`、full-context `.626`，LightGBM test `.427`；与 PDF 一致。 |
| Fig.5 标题/作者提示 | **FAIL** | caption 声称含 “learning curve”，但脚本没有读取 `tier2-learning-curve.json`，图中也没有学习曲线。作者提示又把 prompt-only `.592`（seed17）与 full-context `.637`（接近 seed42）并列，混用了 seed。 |
| Methodology 平台表真值 | **FAIL** | `Qwen` 和 BERT revision 可由 config 验证；但 benchmark GPU 仍是占位值。表中的 PyTorch/vLLM 来自 201 CPU smoke，而 3090 labeling manifest 是 Torch `2.10.0+cu128`、vLLM `0.19.1`，两个环境被混成一张“platform”表。 |
| Gateway revision | **FAIL** | 表中 pin 为 `d49d79d`；本地 `feat/ltr-decision-adapter` 实际 HEAD 已是 `888fba9984a34b23340f08e6faf81ace032f3a01`，且 `d49d79d` 只是其祖先。尚无 final run manifest 证明最终 benchmark 使用哪个 SHA。 |
| Training config/seed | **FAIL** | margin 1.0、delta 0.2、512、batch 16、1 epoch、LR 2e-5 与 config 一致；matrix summary 证明 BERT seeds 17/42/73，但 LightGBM 仅 seed42。表中笼统的 “Replications: Seeds 17, 42, 73” 容易误示所有模型都有三次重复。 |
| Sample manifest | PASS | 6000、seed42、4 桶 `1265/1560/1769/1406`、session-preserving、4000/1000/1000 split 和 SHA 均与 manifest 一致。 |
| `tau=0.642329` 等关键数字 | **FAIL** | `0.6423292768`、LightGBM validation `.446937`/test `.426799`、学习曲线四点和 `2.86×` 都正确；但 `.592/.637` 的 ablation 提示跨 seed，不是公平同口径比较。 |
| References 抽查 5 条 | 部分证据通过，整体 **FAIL** | Fu-LTR/NeurIPS 2024、PARS/preprint 2025、EGTP/ICLR 2026、Agentix/NSDI 2026、vLLM/SOSP 2023 均与 CSV 一致，未发现这 5 条编造 venue/year；但全部缺作者，BERT/LightGBM 条目没有作者、年份或官方 URL，不能算完整 IEEE 引用。 |
| Fig.4/6/7/8 真实性 | **FAIL** | 脚本诚实地拒绝生成缺数据图，没有伪造结果；但对应 measured artifacts/图仍不存在。 |
| AI 生成正文散文 | PASS | PDF 和 section 源中没有任何正文散文段落，只有作者提示注释、表格内容、图标签及 caption。无法机械证明这些非正文元素的作者身份，但不存在违反红线的正文段落。 |

**Ryoo 层判决：FAIL。**

## 必修清单

1. **修正 Fig.2 的图类。**  
   [plot_final_report_figures.py](/Users/alex/develop/vllm-ltr-optimization/scripts/plot_final_report_figures.py:469)、[background.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/sections/background.tex:16)  
   将顺序 lineage 改成静态系统架构：artifact store、training service、checkpoint registry、decision service、gateway、vLLM 等组件及其数据/部署连接；不要表达步骤先后。

2. **把 figure caption 提升到至少 10 pt。**  
   [main.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/main.tex:23)、[appendix_a.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/sections/appendix_a.tex:17)  
   使用 IEEEtran 兼容的 caption 配置覆盖默认约 8 pt 字号，重新编译后用 PDF bbox 再测。

3. **修正 Fig.5 口径。**  
   [evaluation.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/sections/evaluation.tex:14)  
   二选一：  
   - 将 caption 改为 Tier-2 test 与 Tier-1 validation predictor comparison；或  
   - 真正把 learning-curve JSON 接入脚本。  
   Ablation 使用同一 seed（如 seed17 `.592/.642/.617`）或三 seed 均值（`.587/.630/.626`），不得混 seed。

4. **重做平台表的 provenance。**  
   [methodology.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/sections/methodology.tex:16)  
   分开列 training/labeling 与 serving benchmark 环境；最终 GPU、CUDA、Torch、vLLM、model revision、VeloxMesh SHA 必须来自实际 run manifest。将重复说明改成“BERT 17/42/73；LightGBM 42”。

5. **生成四张真实评估图并移除占位状态。**  
   [evaluation.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/sections/evaluation.tex:9)  
   补齐 Fig.4/6/7/8 的 measured input、manifest 与 PDF；没有执行的策略/负载不得画空柱。

6. **修正 float 和精确页数。**  
   [methodology.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/sections/methodology.tex:2)、[evaluation.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/sections/evaluation.tex:2)  
   人工正文加入后确保节标题先于本节表/图，必要时使用 `\FloatBarrier`/float placement 控制；最终逐页量出 ¼、1、1.5、1、1、3、1 页预算，并消除标题页、节标题页和浮动图造成的大空白。

7. **完成真实 bibliography。**  
   [refs.bib](/Users/alex/develop/vllm-ltr-optimization/latex_source/refs.bib:5)、[main.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/main.tex:67)  
   从 CSV/原论文/官方工具来源补齐作者、题名、venue、年份、URL/DOI；为 BERT、LightGBM 使用正式论文或官方引用；正文引用完毕后删除 `\nocite{*}`。

8. **人工完成提交信息和 Appendix。**  
   [main.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/main.tex:43)、[appendix_a.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/sections/appendix_a.tex:10)  
   人工填写真实组号、标题、全体作者；提交源码到 GitHub 后放入真实仓库截图。所有 `[作者手写区]` 必须由学生本人写作。
## 复核一
## 骨架层判决

**仍需修复——暂不授予 `SKELETON-APPROVE`。**

复核对象为 `main @ b41395e4a131f766b45bbf3b018f976b89d4e364`；全程只读，工作区未变化。契约测试 `15 passed`，但仍有 PDF 视觉缺陷和未完成事实被写成已验证的问题。

| 已修项 | 状态 | 独立复核结果 |
|---|---|---|
| 1. Fig. 2 静态组件架构 | **RESOLVED** | 图中为 offline/online 组件、部署边界及数据/接口连接；无编号、步骤序或控制流程语义。 |
| 2. Caption ≥10 pt | **RESOLVED** | 重测 PDF：Fig.1–9、Table I–II caption 字形 bbox 高度全部为 **9.963 pt**，符合 10 pt 字体的实际 PDF bbox。 |
| 3. Fig. 5 数据与三-seed 口径 | **NOT-RESOLVED** | 数据已正确接入：三-seed 均值为 `.586549/.630214/.626129`，显示 `.587/.630/.626`；学习曲线也与 JSON 一致。但 PDF 中 `.605` 标注与 `0.62` y 轴刻度 bbox 实际相交，尚不满足清晰可读；脚本顶部数据源说明也漏列 learning-curve JSON。 |
| 4. 平台 provenance、pin、seeds | **NOT-RESOLVED** | 两阶段拆分、3090 环境、Gateway SHA、BERT `17/42/73` 和 LightGBM `42` 均正确。但 serving run 尚属 pending，表格却写成 “run-manifest verified/recorded”；完整 Gateway SHA 在 PDF 中由单栏边界 `x=300.0` 溢出至 `x=370.9`。 |
| 5. 六条著名引用补全 | **RESOLVED** | vLLM、ToolACE、Toolathlon、BFCL、BERT、LightGBM 已有作者及可核验 publication/project 元数据；小众条目继续保留显式 TODO，没有猜作者。 |

## 必修清单

- [plot_final_report_figures.py](/Users/alex/develop/vllm-ltr-optimization/scripts/plot_final_report_figures.py:1007)：给 learning-curve 首点采用右偏移和左对齐，例如正 `x` offset + `ha="left"`；重画 Fig. 5，确认 `.605` 与 `0.62` bbox 不再相交。
- [plot_final_report_figures.py](/Users/alex/develop/vllm-ltr-optimization/scripts/plot_final_report_figures.py:11)：数据源说明补上 `tier2-learning-curve.json`。
- [methodology.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/sections/methodology.tex:20)：在最终 serving manifest 尚未生成时，改为 “target/pinned; **to be verified** in final run manifest”，不能写 `run-manifest verified`。
- [methodology.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/sections/methodology.tex:23)：同理把 `recorded in run manifest` 改成未来态 `to be recorded`。
- [methodology.tex](/Users/alex/develop/vllm-ltr-optimization/latex_source/sections/methodology.tex:24)：表内使用短 pin（如 `888fba9984a`），完整 SHA 保留在作者注释/正文证据中；重编译后确保内容不越出单栏。

用户指定的四张待生成评估图、正文加入后的页数/float、作者信息与 Appendix 截图仍按约定保持 **pending**，未作为本次拒批理由。
## 复核二
仍需修复（[plot_final_report_figures.py:1013](/Users/alex/develop/vllm-ltr-optimization/scripts/plot_final_report_figures.py:1013)：`.605` 已避开 y 轴，但其 PDF bbox 与 `(n=999)` 注释重叠；需错层/下移后重绘 Fig. 5）。
## 终判
SKELETON-APPROVE
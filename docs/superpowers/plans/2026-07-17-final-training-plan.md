# 最終訓練方案(定稿 2026-07-17)

> 本文取代 `2026-07-16-overnight-training-and-gateway-integration.md`。
> 決策依據:web-ask 兩輪調研(存檔 `~/.claude/web-ask/archive/20260716-2200-*.md`、`20260716-2240-plan-iteration-sol.md`)+ owner 全部拍板。
> **唯一硬 deadline:課程 Final Report 2026-07-27 8:59AM(IEEE 雙欄)+ GitHub repo URL 7/22。IPDPS 不是本階段目標**,凡「10月/IPDPS」項一律不投工時,只進 Discussion 的 future work 文字。

## 0. Scope Freeze(凍結,不再議)

**論文中心命題(novelty 一句話)**:
面向單引擎 agentic LLM serving 的 reliability-and-opportunity-gated length-aware scheduling——系統在請求入隊時聯合估計相對輸出長度與排序可靠性,僅在「預測可信 且 隊列收益超過 predictor 開銷」時啟用 LTR,否則回退 prediction-free tail-safe 調度。核心主張:**predictor 可靠性本身是引擎內調度的顯式控制變量**。

**任務定義(prediction unit)**:
- 預測單元 = 一次即將進入 vLLM engine 的 LLM invocation(agent 任務的每一輪 = 一條樣本)
- Admission-time 可見輸入:當前 prompt + tool schema + 對話歷史 + session/workflow 元數據;禁止用未來 token
- 標籤 = 該輪生成的 completion token 數(reasoning token 計入;工具執行時長不計入)
- 三個可立 claim:①agentic predictor 跨 generator/task 可靠性顯著退化 ②selective confidence 能識別漂移下仍可靠的子集 ③雙 gate 保留 LTR 收益且限制尾部退化

## 1. 已定決策(D1-D9 最終版)

| # | 決策 | 定案 |
|---|---|---|
| D1 | 數據範圍 | **3 個語義源 + 1 個 replay 源**(見 §2),不做六源全量。論文表述:"3 text-rich corpora for training/OOD + production-style traces for replay",絕不寫 "train on six datasets" |
| D2 | authority hardening | 暫停,WIP 存側分支,`__setattr__` 篡改不在 threat model。**全計劃禁止新增 lock/gate/contract/attestation** |
| D3 | 標籤後端 | 兩檔制(§3)。Tier 2 主標籤 = 本地 vLLM Qwen3.5-9B BF16(24G 卡可跑);硅基流動 API(同款)可做 pilot 加速,轉正需本地 1-2k 條一致性驗證 |
| D4 | pilot 門檻 | repo 現有(overall ≤1%,per-stratum ≤3%) |
| D5 | parse_valid | 無權威標籤的 row 排除出分類指標,保留做 length/tau |
| D6 | 數據保存 | 私存快照,只提交 hash/manifest |
| D7 | 預算 | owner 填真值;24G 卡 ~¥1.5-2/h(標籤+訓練),48G 只租 benchmark 那 1-2 天 |
| D8 | predictor 陣容 | 9+1(§4)。舊聊天 5 checkpoint 做 zero-shot 遷移對照(只推理) |
| D9 | 被服務模型 | **Qwen/Qwen3.5-9B BF16 單卡**(選型理由見 §6)。Gemma 4 12B / gpt-oss-20b / Qwen3.6 雙卡全部延後,不進 7 月 |

## 2. 數據源(角色制,不是六源平等)

| 源 | 角色 | 備註 |
|---|---|---|
| **toolace** | 語義訓練 + schema 多樣性 | ICLR'25,pinned |
| **toolathlon** | 語義訓練/長程 workflow | ICLR'26([arXiv 2510.25726]),17 模型×3 runs 軌跡 |
| **bfcl** | held-out OOD 測試(永不訓練) | 標準 benchmark |
| **lmcache(canonical)** | session/arrival/KV **replay** | ⚠️ 換源:`DiscoPosse/...` 是轉載,改用原始 **`sammshen/lmcache-agentic-traces`**(787 sessions, 24,881 iterations),釘 revision |
| semianalysis / inferact | 暫緩(解析計劃保留,不做 production loader) | 7 月不投工時 |
| nvidia/Open-SWE-Traces | 未來 OOD corpus(不替換 lmcache) | 20 萬合成 SFT 軌跡,角色不同 |

Split 規則:按 session / task template / tool schema 劃分,禁止按單條 prompt 隨機切(防模板洩漏)。跨源 MinHash 去重保留(已有)。

## 3. 標籤(兩檔制)

**Tier 1 — observational trace outcomes(弱監督,免費,先行)**
- 直接讀軌跡錄制長度(lmcache 有現成 `output_length`;其他源用 tokenizer 數錄制的 assistant 回覆)
- 必存 generator ID + decoding 元數據;**不同 generator 不盲混**
- 用途:workload characterization、predictor 弱預訓練、cross-generator transfer 研究
- 論文表述:heterogeneous observational outcomes,**不是** served model 的 ground truth

**Tier 2 — target-model replay labels(主標籤)**
- 釘死:`Qwen/Qwen3.5-9B` revision `c202236…` + vLLM commit + chat template + `qwen3_coder` tool parser + `enable_thinking:false` + greedy(temp=0)+ **max_tokens=4096**(從 2048 上調)
- 截斷樣本 = right-censored:打 `censored` 標記,報告每源 censor rate,做 2048 vs 4096 敏感性表;主 loss 先排除 censored
- 機制 = 錄像回放 + teacher forcing:工具結果用軌跡錄制的,不真執行工具;Qwen 每樣本只生成一輪
- 訓練對比三版:T2 only / T1 預訓練→T2 微調 / T1 預訓練→T2 只校準(若後兩者不優於第一,Tier 1 只留 characterization)

## 4. Predictor 訓練矩陣(9+1,取代 15-run)

| 變體 | seeds |
|---|---|
| prompt only | 17/42/73 |
| prompt + tool schema | 17/42/73 |
| full admission context(+history+workflow)| 17/42/73 |
| LightGBM/QRF 結構特徵基線(input長度/schema數/turn數等) | 1 |

- 砍掉獨立 +history、+workflow 變體和 3 個 diagnostic(midterm 已證增益主要在 backbone,細粒度特徵工程收益小)
- 3-seed ensemble disagreement = confidence 信號;產出 selective risk-coverage 曲線
- 硬件:24G 卡即可(BERT 110M);Tier 1 標籤下今日即可開訓

## 5. 測試四層(scope 修訂)

| 層 | 內容 | 7 月做? |
|---|---|---|
| 1 predictor 離線 | tau/MAE/calibration/OOD,ID→generator shift→schema shift | ✅ 全做 |
| 2 CPU 模擬器 | 全策略×10 seeds×閾值掃描;**主結果改非搶佔式**(免費搶佔只放 sensitivity);用實測 tau 替換 tau_synth | ✅ 全做(純 CPU)|
| 3 真機 benchmark | 48G 卡 1-2 天:負載按 saturation 40/70/90%+burst;全負載只跑 FCFS/pure-LTR/tail-safe/gated 四策略;oracle/random/aging 只在一個高負載點;**先跑 stock-FCFS vs custom-shim parity(預定義 3-5% 容差)**;predictor CPU 延遲計入端到端;主指標 TTLT/normalized slowdown/P95/P99(不是 TTFT);3 次重複+CI | ✅ 縮減版 |
| 4 VeloxMesh E2E | **7 月只放 architecture diagram + integration stub**,真 E2E 延後 | ⏸️ |

## 6. 被服務模型選型理由(報告 Methodology 素材)

1. 單卡 BF16 裝得下且留足 KV 池(權重≈18G;顯存公式:GB≈參數B×2+2~3G 雜項,其餘全給 KV;agentic 長上下文請求一條 KV 0.6-1.4G+,並發實驗吃的就是這個池)
2. vLLM 部署鏈路官方成熟(model card 直給 qwen3 reasoning parser + qwen3_coder tool parser 配置)——改調度器的研究不能同時跟部署打架
3. 可釘 revision + 本地 BF16 → 標籤與 benchmark 同 checkpoint,逐字節可復現
4. 2026-02 發布,開源單卡檔最新一代;更大模型(Qwen3.6-35B+/Kimi/DeepSeek V4/MiniMax)總參數塞不進單卡 BF16(MoE active 參數 ≠ 顯存)
5. thinking mode 可關 → greedy 標籤穩定
6. 量化紀律:主實驗全程 BF16;量化只在「同配置自比 + 披露」下才合法,7 月不用

## 7. 報告六圖(卡死 Evaluation ≥6 圖要求)

1. FCFS vs LTR 真機復現(已有,midterm)——動機
2. predictor ranking gap:chat(LMSYS→ShareGPT,已有)+ agentic ID/OOD(新)
3. 特徵消融:3 變體 + 結構基線(9+1 產出)
4. selective risk-coverage 曲線(confidence 質量)
5. 模擬器 reliability envelope heatmap(何時信、何時退)
6. 真機 load sweep:四策略 P95/P99 TTLT(計入 predictor 開銷)

全部 matplotlib,腳本入 `scripts/`,repo 需 `latex_source/`。**AI 禁代寫正文**——AI 只做實驗/數據/畫圖腳本,文字 Alex 自寫。

## 8. 執行時間線

**今日(7/17)**:
1. Codex:數據源重分類 + canonical lmcache 換源(釘 revision)
2. Codex:打通 **toolace loader**(一個,production 級)
3. Codex:Tier 1 標籤提取器(數錄制長度 + generator ID)→ **當天開訓 BERT 第一個變體**(24G 卡或本地)
4. Codex:stock-FCFS vs custom-shim parity harness(防代碼路徑質疑)
5. Alex:租 24G 卡;(可選)硅基流動 API key 備用

Stop/go:censor 率異常高→max_tokens 提 8192;tool-call parse 失敗率高→先修 template 再批量;parity 超容差→先修 scheduler 路徑。

**本週(7/18-21)**:toolathlon + bfcl loader → Tier 2 pilot 300-500 條(看長度分布/censor率/吞吐)→ Tier 2 全量標籤 → 9+1 訓練完 → 模擬器全量重跑(實測 tau)→ 圖 2-5 出爐。7/21 repo 清理 + collaborator。

**7/22**:交 GitHub repo URL(Public 或加 `anithasaravanaedu-spec`,點 Submit)。

**7/22-24**:租 48G 跑第 3 層 benchmark → 圖 6。LaTeX 骨架 + `scripts/` 齊。

**7/24-26**:Alex 寫正文(AI 不碰),逐條對 rubric 二元自查(頁數/圖數/新頁/字號)。

**7/27 8:59AM 前**:交 PDF。

## 9. 禁做清單(誰提都不做,7 月內)

六源全量 loader / 15-run 矩陣 / Qwen3.6 雙卡 / VeloxMesh 真 E2E / API 旗艦實驗 / Tier 1 盲混當 ground truth / 免費搶佔模擬器當主結果 / TTFT 當主指標 / 任何新 lock/gate/contract / 追 256K max-model-len(按 workload CDF 選 32K 級)

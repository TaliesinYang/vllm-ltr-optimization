# 剩餘工作清單(2026-07-18 下午)

> 唯一硬 deadline:repo URL 7/22 · Final Report 7/27 8:59AM(IEEE 雙欄,Eval 3頁≥6圖,AI 禁代寫正文)
> 已完成:訓練(tau 0.642)· 合併 · 真 BERT 接入 /v1/decision · gRPC 包壳(能接他 gateway)· 數據全上 OSS

## A. 六張報告圖 —— 缺哪些、要什麼

| 圖 | 內容 | 數據來源 | 狀態 |
|---|---|---|---|
| Fig.1 | FCFS vs LTR 復現(2.86× TTFT)| midterm 已測 | ✅ 有 |
| Fig.2 | predictor 泛化 gap:chat(已有)+ agentic ID/OOD(新)| 需 BERT 在 bfcl/toolathlon 等外部源打分 | ⬜ 缺外部源評估 |
| Fig.3 | 特徵消融 + 結構基線 | tier2 矩陣 tau(已有 0.64/0.47)| ✅ 有數據,待畫 |
| Fig.4 | selective risk-coverage 曲線 | 需 BERT 測試集逐條打分 + confidence | ⬜ 缺逐條打分 |
| Fig.5 | 模擬器 reliability envelope(實測 tau 校準)| 需真打分向量驅動模擬器 | ⬜ 缺逐條打分 + 接線 |
| Fig.6 | 全鏈路 load sweep(四策略 P95/P99 TTLT)| **需 GPU 租卡 + gateway benchmark** | ⬜ 周末 |

**關鍵依賴:Fig.2/4/5 都卡在同一件事——「BERT 逐條打分」還沒生成。** 生成一次,三張圖的料都有。

## B. 剩餘工作(按依賴排序)

### B1. 生成 predictor 逐條打分(CPU,~5分鐘)—— 解鎖 Fig.2/4/5
- 跑 BERT(checkpoints_best_predictor)在:①1000 條測試集 ②外部 OOD 源(bfcl/toolathlon 抽樣)
- 存逐條:{request_id, rank_score, confidence, true_length}
- 產出:test-scores.jsonl + ood-scores.jsonl

### B2. 模擬器重跑(CPU,~10分鐘)—— 出 Fig.5
- 用 B1 的真打分向量(帶錯誤結構)驅動 gateway_policy_probe 模擬器
- 掃負載 × 策略,出 reliability envelope,標出實測工作點 0.642
- 產出:Fig.5 數據 + envelope heatmap

### B3. 畫圖腳本(CPU)—— Fig.2/3/4/5 出圖
- matplotlib,入 scripts/,每圖可復現(教授硬要求)
- Fig.3 現在就能畫(數據齊);Fig.2/4/5 等 B1/B2

### B4. GPU benchmark(租卡,周末,~半天)—— 出 Fig.6
- 租 48G → 起 vLLM(Qwen3.5-9B)+ gateway(fc20873 或他 main + gRPC 壳)
- 全鏈路四策略 × 負載 × 3 重複 → TTLT/P95/P99
- 前置:FCFS parity + gateway 開銷基線
- 產出:Fig.6 數據

### B5. 代碼收尾(必須,7/22 前)
- push main(現 ahead origin 36 commit)
- repo 清理:latex_source/ + scripts/(每圖腳本)+ 加 collaborator anithasaravanaedu-spec
- 7/22 交 repo URL

### B6. 報告寫作(7/22-26,Alex 自寫,AI 禁碰正文)
- LaTeX IEEE 骨架(我可搭)+ 9 節 + 頁數卡死
- 隊友 Mingye/Yibo 各自小節 7/24 交
- 正文 Alex 寫,rubric 逐項自查

### B7. 團隊協作(今天發)
- Mingye/Yibo 消息:分工 + 7/24 交稿線 + gRPC 整合對齊(你的 BERT = predictor,他 = serving 基建)

## C. 我做 / 你做 / Codex 做

| 誰 | 做什麼 |
|---|---|
| **我(Claude)** | B1 逐條打分 · B2 模擬器 · B3 畫圖 · B6 LaTeX 骨架(不碰正文)· 稽核 Codex |
| **Codex** | (可選)B1/B2 若你想它做;gRPC 壳已完成 |
| **你(Alex)** | 租卡(B4)· push+repo 清理(B5)· 寫正文(B6)· 發團隊消息(B7)· 所有對外決策 |

## D. 現在的岔路(今天能推進的,都不占 GPU)

1. **B1+B2+B3(Fig.2/4/5 一條龍)** ← 我建議現在做,解鎖三張圖
2. push main + repo 清理(B5)← 攢了 36 commit,該推
3. 團隊消息(B7)← 隊友交稿倒計時

GPU 相關(B4/Fig.6)= 周末租卡,今天做不了。
報告正文(B6)= 圖齊之後,你寫。

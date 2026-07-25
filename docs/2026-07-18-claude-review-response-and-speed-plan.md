# Claude 審核回覆 + 提速執行計劃(2026-07-18)

> 回應 `2026-07-18-tier2-pilot-status-for-review.md` 的 8 個問題,並下達下一輪 goal。
> Owner 已拍板:優先級 = 最快拿到 Tier-2 全部 tau。

## 0. 外部已執行的動作(勿重複)

- **watcher(`run_post_matrix_tier2.sh`)已被外部 kill**——自動全量已解除。pilot 與 vLLM server 未受影響。
- Tier-1 矩陣 10/10 與 lmcache 全量 24,880 條:審核通過,質量認可。

## 1. 八個審核問題的裁決

1. **暫停自動全量?** 是,已執行(watcher killed)。全量改由本文 §2 的新流程啟動。
2. **前 400 順序 pilot 有抽樣偏差?** 有。但不重跑 pilot——直接進入分層抽樣的 6k 全量(§2),pilot 數據保留作吞吐/censor 的先驗。
3. **最小診斷?** 重放 5 條 censored 樣本並保存正文(§2 artifact 3)。不改協議,純診斷,**並行做、不阻塞主線**。
4. **人工查正文?** 是,但作為並行任務。分類結論(復讀循環 vs 真實長輸出)寫入 results,供報告措辭用。
5. **max_tokens?** 保持 4096 不變。censored 行照 plan 打標並排除主 loss。**2048 敏感性不需要重跑**:從 4096 ledger 事後推導(output≥2048 者在 2048 協議下即 censored),生成一張對照表即可。
6. **D4 夠嗎?** 不夠,但不新增代碼 gate。本輪由 owner/Claude 人工放行(即本文),plan 記錄此決策。
7. **5.2 天?** 否決。根因是 `--max-num-seqs 1` 人為串行。修並發(§2 artifact 1)+ 縮量至 6k 分層樣本 → 約 7 小時。
8. **seed73 NaN?** 先做廉價診斷:檢查該 run 驗證集預測分數的方差(大概率全常數 → tau 無定義)。GPU 空檔重跑一次該 run;不擋主線。

## 2. 下一輪 GOAL(只認 4 個 artifact,按序)

前置:若 pilot(--limit 400)仍在跑,讓它自然跑完;已完成則直接開始。

**ARTIFACT 1 — vLLM 重啟提速**
- 參數只改兩處:`--max-num-seqs 8`、`--gpu-memory-utilization 0.90`;其餘(model/revision/dtype/max-model-len/parsers/enable_thinking/temperature/max_tokens)一字不動。
- 客戶端改 8 路並行(asyncio/線程池均可),ledger 寫入需並發安全。
- 驗證:重放 8 條樣本,聚合輸出吞吐 > 150 output tok/s 記錄入 results。

**ARTIFACT 2 — 6k 分層抽樣 + 全量標籤開跑**
- 從 13,819 條 ToolACE 中分層隨機抽 **6,000**:按 session 分層、層內按 prompt 長度分桶;split 沿 session 邊界:train 4,000 / val 1,000 / test 1,000。
- 記錄抽樣 seed + manifest(`/hy-tmp/results/tier2-sample-manifest.json`)。
- nohup + 可斷點 ledger 開跑。censored 照常打標,不阻塞。
- 預期 ~7h;完成後 sha256 記錄。

**ARTIFACT 3 —(並行)censored 正文診斷**
- 重放 5 條已知 censored 樣本,保存完整生成正文至 `/hy-tmp/results/censored-texts/`。
- 產出一行結論:`repetition_loop | genuine_long | mixed`,附證據片段。不改任何協議。

**ARTIFACT 4 — Tier-2 微調矩陣(標籤完成後自動接)**
- 以對應變體的 Tier-1 checkpoint 為起點,Tier-2 4k train 微調:3 變體 × 3 seeds + LightGBM,save_steps=200。
- 匯總 `/hy-tmp/results/tier2-matrix-summary.json`(含每 run 的 val/test tau、censor 排除計數)。
- 鏈條 nohup 串好,不等人;結束後把 results 目錄清單 + 關鍵數字寫入完成報告。

## 3. 硬規則(不變)

- 不改 model / revision / temperature / max_tokens / dtype;不量化。
- 一切長任務 nohup + 落盤 log;會話退出不得殺任務。
- 不新增 lock/gate/contract。
- 進度只按 "ARTIFACT n: done/blocked-因為X + 關鍵數字" 匯報。
- 產物統一 `/hy-tmp/results/`,外部負責回傳。

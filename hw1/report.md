---
title: Machine Learning Assignment 1
author: 409410005 鍾天睿
---
# Machine Learning Assignment 1

## Execution Description

`@timer_and_result`是用來計時以及顯示 return value 的裝飾器。`pla` 執行 PLA 回傳執行步數，`pocket_pla` 最多執行 30 輪，回傳最好的結果。更新參數的方法參考簡報。

## Experimental Results

```
============= problem 2 =============
pla                  0.0003s, epoch: 1, step: 29

============= problem 3 =============
pla                  0.0095s, epoch: 1, step: 1999
pocket_pla           0.0017s, epoch: 0, step: 155, correct_cnt: 2000

============= problem 4 =============
pocket_pla           0.3572s, epoch: 29, step: 1999, correct_cnt: 1661
```

只有50個點的問題二很快速就收斂，只用了 1 個 epoch 就更新完成（最後一個 epoch 相當於檢查，epoch 從 0 開始）。問題3 pocket PLA 收斂較快，即使每個 step 都檢查所有的點的預測，仍然比單純使用 PLA 快。

問題4的準確度為 1661/2000 = 0.8305，標記的正確率為 1900/2000，距離標記的正確率有一點距離。

## Conclusion

當有 mislabel 時，pocket PLA 無法全部正確，只能等到轉確度達到接受或是迴圈結束。

有 mislabelled data 時，Pocket PLA 的準確度較低。

## Discussion

> The questions or the difficulties you met during the implementation:

不會 很簡單。

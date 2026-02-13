import pandas as pd
import os
import numpy as np
import scipy
from scipy import stats
import random
import json
import torch
import torch.nn.functional as F

SPECIAL_S_IDS = {"S001", "S006", "S013", "S017", "S018", "S033"}    # DEMO system use special prompts list

def get_texts_from_filename(data_dir, filenames):
    prompt_ids = []
    system_ids = []
    for fn in filenames:     # audiomos2025-track1-S002_P044.wav
        fn = fn.replace("audiomos2025-track1-","")
        s_id = fn.split("_")[0]
        p_id = fn.split("_")[1].split(".")[0]
        system_ids.append(s_id)
        prompt_ids.append(p_id)

    df = pd.read_csv(f'{data_dir}/prompt_info.txt', sep='	')
    demo_df = pd.read_csv(f'{data_dir}/demo_prompt_info.txt', sep='	')
    texts = []
    for s_id, p_id in zip(system_ids, prompt_ids):
        if s_id in SPECIAL_S_IDS:   # demo_prompt_info
            demo_id = 'audiomos2025-track1-' + s_id + '_' + p_id + '.wav'
            text = demo_df.loc[demo_df['id'] == demo_id, 'text'].values
        else:   # prompt_info
            text = df.loc[df['id'] == p_id, 'text'].values
        
        if len(text) > 0:
            texts.append(text[0])
        else:
            texts.append(None)
    return texts

def compute_metrics(y_true, y_pred):
    # Ensure inputs are numpy arrays
    y_true_np = np.array(y_true).squeeze() # .squeeze() to remove any single dimensions like (N,1) -> (N,)
    y_pred_np = np.array(y_pred).squeeze()

    # Check for empty or insufficient data after conversion
    if y_true_np.size == 0 or y_pred_np.size == 0 or len(y_true_np) < 2: # Check .size for numpy arrays
        print("compute_metrics received empty or insufficient data. Returning NaNs.")
        return np.nan, np.nan, np.nan, np.nan

    mse  = np.mean((y_true_np - y_pred_np)**2) # Use the numpy arrays
    
    if np.std(y_true_np) == 0 or np.std(y_pred_np) == 0:
        # If one has stddev 0 and the other doesn't, LCC is undefined (NaN).
        # If both have stddev 0, they are constant. If they are the same constant, LCC could be 1 (perfectly correlated),
        # but np.corrcoef might return NaN or raise warning. Let's return NaN to be safe if any std is 0.
        lcc = np.nan
    else:
        lcc  = np.corrcoef(y_true_np, y_pred_np)[0,1]
    
    try:
        srcc = stats.spearmanr(y_true_np, y_pred_np).correlation
    except (ValueError, TypeError, ZeroDivisionError): # Added ZeroDivisionError for robustness
        srcc = np.nan
    try:
        ktau = stats.kendalltau(y_true_np, y_pred_np).correlation
    except (ValueError, TypeError, ZeroDivisionError): # Added ZeroDivisionError for robustness
        ktau = np.nan
        
    return mse, lcc, srcc, ktau

# systemID function as provided in Yi-Cheng branch
def systemID(wavID):
    try:
        # Example wavID: "audiomos2025-track1-S002_P044.wav"
        return wavID.replace("audiomos2025-track1-","").split('_')[0]
    except Exception as e:
        # print(f"Error parsing system ID from {wavID}: {e}") # for debugging
        return "unknown_system" # return a placeholder

# === [新增] 計算一個 Batch 內所有樣本兩兩配對的 Pairwise Ranking Loss ===
def compute_pairwise_ranking_loss(pred_scores, true_scores, margin=0.0, device='cuda'):
    """
    計算一個 Batch 內所有樣本兩兩配對的 Ranking Loss。
    
    Args:
        pred_scores (Tensor): 模型預測的純量分數 (B, 1) 或 (B,)
        true_scores (Tensor): 真實的 MOS 分數 (B, 1) 或 (B,)
        margin (float): MarginRankingLoss 的邊界值 (預設 0.0 或 0.1)
        device (str): 運算設備
        
    Returns:
        loss (Tensor): 計算出的 scalar loss
    """
    # 1. 確保形狀為一維向量 (Batch_Size,)
    pred = pred_scores.view(-1)
    true = true_scores.view(-1)
    batch_size = pred.size(0)

    # 2. 建立 Pairwise 比較矩陣
    # diff_true[i, j] = true[i] - true[j]
    # 利用廣播機制: (B, 1) - (1, B) -> (B, B)
    diff_true = true.unsqueeze(1) - true.unsqueeze(0)

    # 3. 產生目標標籤 (Targets)
    # 1: i > j (i 比較好)
    # -1: i < j (j 比較好)
    # 0: i == j (分數相同)
    targets = torch.sign(diff_true)

    # 4. 建立遮罩 (Mask)
    # 我們只關心分數「不同」的配對，忽略分數相同的 (targets=0) 以及自己比自己
    mask = (targets != 0)

    # 如果這個 Batch 裡大家分數都一樣 (極端情況)，直接回傳 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 5. 準備輸入給 MarginRankingLoss 的資料
    # 擴展預測值成 (B, B) 矩陣，以便與 targets 對應
    pred_i = pred.unsqueeze(1).expand(batch_size, batch_size)
    pred_j = pred.unsqueeze(0).expand(batch_size, batch_size)

    # 6. 利用 Mask 篩選出有效的配對 (Flatten)
    p_i_flat = pred_i[mask]
    p_j_flat = pred_j[mask]
    t_flat = targets[mask]

    # 7. 計算損失
    # Loss = max(0, -target * (input1 - input2) + margin)
    loss = F.margin_ranking_loss(p_i_flat, p_j_flat, t_flat, margin=margin, reduction='mean')
    
    return loss

# === [新增] 計算一個 Batch 內所有樣本的 Listwise Ranking Loss (基於 ListNet) ===
def compute_listwise_ranking_loss(pred_scores, true_scores, temperature=1.0, device='cuda'):
    """
    計算一個 Batch 內所有樣本的 Listwise Ranking Loss。
    
    核心概念 (ListNet)：
    把整個 Batch 視為一個 List，將「預測分數」與「真實分數」通過 Softmax 轉換為機率分佈。
    分數越高，佔據的機率質量就越大。
    然後計算這兩個分佈之間的 Cross Entropy (等價於 KL Divergence)。
    這能強迫模型學習整個 Batch 的全域排序 (Global Ranking)。
    
    Args:
        pred_scores (Tensor): 模型預測的純量分數 (B, 1) 或 (B,)
        true_scores (Tensor): 真實的 MOS 分數 (B, 1) 或 (B,)
        temperature (float): Softmax 溫度參數，用於平滑或銳化分佈。
                             MOS 分數通常很密集，適當調整 T 可以放大差異 (預設 1.0)
        device (str): 運算設備
        
    Returns:
        loss (Tensor): 計算出的 scalar loss
    """
    # 1. 確保形狀為一維向量 (Batch_Size,)
    pred = pred_scores.view(-1)
    true = true_scores.view(-1)
    
    # 防呆：如果 Batch 只有一筆資料，無法計算排列，直接回傳 0
    if pred.size(0) <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 2. 將分數轉換為機率分佈 (Softmax)
    # 真實分數的分佈 (Target Distribution)
    # 分數越高，機率越大。使用 detach() 確保真實分數不參與梯度計算。
    true_dist = F.softmax(true / temperature, dim=0).detach()
    
    # 預測分數的分佈 (Predicted Distribution)
    # 使用 log_softmax 是因為後面算 Cross Entropy 時數值更穩定，能避免 log(0)
    pred_log_dist = F.log_softmax(pred / temperature, dim=0)

    # 3. 計算 Listwise Loss (Cross Entropy)
    # 公式: Loss = - sum(P_true * log(P_pred))
    # 目的: 讓模型預測的排序分佈逼近真實的排序分佈
    loss = -torch.sum(true_dist * pred_log_dist)
    
    return loss

#!/bin/bash

# ==========================================
# 環境與路徑設定
# ==========================================
export CUDA_VISIBLE_DEVICES="0"

# 資料集根目錄
DATA_DIR="/share/nas169/jethrowang/MusicEval-full"

# ⚠️ 注意：您的程式碼中這三個 list 參數設為 required=True
# 即使是純評估模式，也必須給予路徑才不會跳 argparse 錯誤
TRAIN_LIST="${DATA_DIR}/sets/train_mos_list.txt"      # 僅為滿足 argparse，評估時不會讀取
VALID_LIST="${DATA_DIR}/sets/dev_mos_list.txt"        # 僅為滿足 argparse，評估時不會讀取
TEST_LIST="${DATA_DIR}/sets/test_mos_list.txt"        # 您真正要進行評估的檔案清單 (如 eval_list.txt 或 test_mos_list.txt)

# ==========================================
# 模型與實驗設定
# ==========================================
# 實驗名稱與目標 checkpoint 路徑
EXP_NAME="primary_model_gaussian_pairwise_ranking_loss"
CKPT_PATH="/share/nas169/jethrowang/DORA-MOS/exp/${EXP_NAME}/ckpt_best_val_combined.pth"

# 根據您的路徑名稱 (gaussian_pairwise_ranking_loss) 推測的設定
# 若您訓練時的 model_type 不同 (例如 muq_roberta_transformer_decoupled_dist)，請務必修改下方變數
MODEL_TYPE="muq_roberta_transformer_dist" 
SCORE_STYLE="gaussian"
RANKING_LOSS_TYPE="pairwise"

# ==========================================
# 執行 Python 腳本
# ==========================================
echo "🚀 開始載入 Checkpoint 進行預測 (Predict Only 模式)..."
echo "📂 Checkpoint: ${CKPT_PATH}"
echo "📄 測試名單: ${TEST_LIST}"

python your_script_name.py \
    --datadir "$DATA_DIR" \
    --expname "$EXP_NAME" \
    --model_type "$MODEL_TYPE" \
    --train_list_path "$TRAIN_LIST" \
    --validation_list_path "$VALID_LIST" \
    --test_list_path "$TEST_LIST" \
    --predict_only_ckpt_path "$CKPT_PATH" \
    --predict_output_filename_base "answer" \
    --dist_prediction_score_style "$SCORE_STYLE" \
    --use_ranking_loss \
    --ranking_loss_type "$RANKING_LOSS_TYPE" \
    --valid_batch_size 16

echo "✅ 評估完成！請至 exp/${EXP_NAME} 資料夾下查看 answer.txt 以及 detailed_preds_*.pt 檔案。"
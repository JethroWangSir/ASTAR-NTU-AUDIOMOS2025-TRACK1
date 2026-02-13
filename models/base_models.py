import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer
from models.base import BasePredictor
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000): # max_len needs to cover longest sequence
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Becomes (max_len, 1, d_model)
        self.register_buffer('pe', pe) # Not a model parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        if x.dim() == 3 and x.size(1) == self.pe.size(1): # if batch_first=False (seq_len, batch, dim)
            x = x + self.pe[:x.size(0), :]
        elif x.dim() == 3 and x.size(0) == self.pe.size(1): # if batch_first=True (batch, seq_len, dim)
            x = x + self.pe[:x.size(1), :].transpose(0,1)
        else:
            # Fallback for 2D tensor (e.g. CLS token) - no positional encoding needed or apply differently
             pass # Or raise error if unexpected
        return self.dropout(x)

class AttentivePooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional tensor of shape (batch_size, seq_len) for padding
        Returns:
            Tensor of shape (batch_size, input_dim)
        """
        # Calculate attention scores
        # (batch_size, seq_len, input_dim) -> (batch_size, seq_len, 1)
        attention_scores = self.attention_weights(x)

        if mask is not None:
            # Apply mask (set scores for padding tokens to a very small number)
            attention_scores.masked_fill_(mask.unsqueeze(-1) == 0, -1e9) # Mask out padding

        # Softmax to get probabilities
        attention_probs = F.softmax(attention_scores, dim=1)

        # Weighted sum
        # (batch_size, seq_len, 1) * (batch_size, seq_len, input_dim) -> sum over seq_len
        context = torch.sum(attention_probs * x, dim=1)
        return context

class BaseTransformerPredictor(BasePredictor):
    """
    Base class for the dual-branch transformer architecture.
    Handles feature extraction, temporal modeling, and fusion.
    """
    def __init__(self, muq_model, roberta_model,
                 audio_transformer_layers=1, audio_transformer_heads=4, audio_transformer_dim=1024,
                 common_embed_dim=768, cross_attention_heads=4, dropout_rate=0.3):
        super().__init__()
        self.muq = muq_model
        self.roberta = roberta_model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        for param in self.muq.parameters():
            param.requires_grad = False
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.muq_output_dim = 1024  # MuQ-large
        self.roberta_output_dim = 768 # RoBERTa-base
        self.common_embed_dim = common_embed_dim

        # --- Audio Path (Temporal Modeling) ---
        self.audio_pos_encoder = PositionalEncoding(d_model=self.muq_output_dim, dropout=dropout_rate)
        
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.muq_output_dim, nhead=audio_transformer_heads,
            dim_feedforward=audio_transformer_dim * 2, dropout=dropout_rate,
            activation='relu', batch_first=True
        )
        self.audio_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=audio_encoder_layer, num_layers=audio_transformer_layers
        )
        # Pooling for MI branch
        self.audio_attentive_pool = AttentivePooling(input_dim=self.muq_output_dim)

        # --- Feature Fusion (Cross-Attention) ---
        # Projections to common space (Linear layers in diagram)
        self.audio_seq_proj = nn.Linear(self.muq_output_dim, self.common_embed_dim)
        self.text_seq_proj = nn.Linear(self.roberta_output_dim, self.common_embed_dim)

        # Cross-Attention (Feature Fusion in diagram)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.common_embed_dim, num_heads=cross_attention_heads,
            dropout=dropout_rate, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(self.common_embed_dim)
        
        # Pooling for TA branch
        self.fused_attentive_pool = AttentivePooling(input_dim=self.common_embed_dim)

    def forward_features(self, wavs, texts, use_decoupled_audio_for_cross_attn=False):
        """
        Forward pass for feature extraction and fusion.
        """
        # --- Base Feature Extraction (Frozen Encoders) ---
        # Encoders are frozen in __init__. We set them to eval() mode.
        self.muq.eval()
        self.roberta.eval()
            
        # We do not need torch.no_grad() here if requires_grad=False is set correctly. Please check if this works, this refactored code has not been verified.
        muq_output = self.muq(wavs, output_hidden_states=False)
        audio_seq_embed_raw = muq_output.last_hidden_state # (B, T_a, D_a)

        # Text Encoder
        text_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
        text_attention_mask = text_inputs['attention_mask'].to(wavs.device) # (B, T_t)
        text_inputs_on_device = {k: v.to(wavs.device) for k, v in text_inputs.items()}
        roberta_output = self.roberta(**text_inputs_on_device)
        text_seq_embed = roberta_output.last_hidden_state # (B, T_t, D_t)

        # Note: Audio padding mask is currently not utilized in the original implementation.
        audio_padding_mask = None # Transformer expects True where padded

        # --- Audio Path (Temporal Modeling) ---
        audio_seq_embed_pe = self.audio_pos_encoder(audio_seq_embed_raw)
        audio_transformed = self.audio_transformer_encoder(
            src=audio_seq_embed_pe,
            src_key_padding_mask=audio_padding_mask
        )

        # Pooling for MI branch
        pooled_audio_features = self.audio_attentive_pool(
            audio_transformed,
            mask=audio_padding_mask
        )

        
        # Determine audio input for cross-attention (Key, Value)
        if use_decoupled_audio_for_cross_attn:
            audio_input_for_fusion = audio_seq_embed_raw # Decoupled Architecture
        else:
            audio_input_for_fusion = audio_transformed # Standard Architecture

        # Project to common dimension
        audio_seq_proj = self.audio_seq_proj(audio_input_for_fusion)
        text_seq_proj = self.text_seq_proj(text_seq_embed)

        # Text padding mask (True where PADDED)
        text_padding_mask = (text_attention_mask == 0)

        # Cross-attention: Text (Q) attends to Audio (K, V)
        cross_attended_output, _ = self.cross_attention(
            query=text_seq_proj, key=audio_seq_proj, value=audio_seq_proj,
            key_padding_mask=audio_padding_mask
        )

        cross_attended_output_norm = self.cross_attention_norm(cross_attended_output)

        # Pooling for TA branch
        fused_features = self.fused_attentive_pool(
            cross_attended_output_norm,
            mask=text_padding_mask
        )

        return pooled_audio_features, fused_features

# class SemanticGuidedChannelInjection(nn.Module):
#     """
#     Novelty:
#     Instead of standard Cross-Attention, we use Text to dynamically re-calibrate
#     Audio channels. This addresses the 'Ambiguity' pain point where vague text
#     should guide WHICH features (channels) to focus on, rather than WHERE (time) to look.
#     """
#     def __init__(self, audio_dim, text_dim, reduction=4):
#         super().__init__()
        
#         # 1. Text Semantic Extractor (Global & Local)
#         # 不只看個別詞，也看整句的語義 Summary
#         # 這裡使用 AttentivePooling，因為一句話裡只有少數關鍵詞（如 "noisy", "muffled"）重要，
#         # 冠詞介系詞（"the", "is"）不重要。Attention 機制會自動抓出關鍵詞。
#         self.text_pool = AttentivePooling(text_dim) 
        
#         # 2. Channel Attention Generator (Squeeze-and-Excitation style but cross-modal)
#         # 用文字特徵生成 Audio 的 Channel Weights
#         # 目的：學習「文字語義」跟「音訊通道」之間的非線性關係。
#         # reduction=4 是為了減少參數量，形成一個 Bottleneck，強迫模型學習最精華的特徵對應。
#         self.se_fc = nn.Sequential(
#             nn.Linear(text_dim, audio_dim // reduction),
#             nn.ReLU(),
#             nn.Linear(audio_dim // reduction, audio_dim),
#             nn.Sigmoid() # 輸出 0~1 的權重
#         )

#         # 3. Injection / Fusion Layer
#         # 為了保留原始音訊細節，我們使用 Residual 結構
#         self.out_proj = nn.Linear(audio_dim, audio_dim)
#         self.norm = nn.LayerNorm(audio_dim)

#     def forward(self, audio_seq, text_seq, text_mask=None):
#         """
#         audio_seq: (B, T_a, D_a)
#         text_seq:  (B, T_t, D_t)
#         text_mask: (B, T_t) True where PADDED (reverse of standard mask usually)
#         """
#         # Step 1: Summarize Text into a Global Semantic Vector
#         # 我們假設文字的順序在 '指導音訊特徵' 上沒那麼重要，重要的是'關鍵詞'
#         # text_mask: True where padded. AttentivePooling expects mask=True where valid? 
#         # 需確認 AttentivePooling 的 mask 定義，通常是 True where Valid (0 in padding mask)
#         # 這裡假設傳入的是 boolean mask (True=Padding)，需要反轉
#         # 把這句話從序列 (32, 20, 768) 濃縮成一個語義向量 (32, 768)
#         # 這代表了整句話的 "Global Context" (全域語義)
#         valid_mask = ~text_mask if text_mask is not None else None
#         text_global = self.text_pool(text_seq, mask=valid_mask) # (B, D_t)

#         # Step 2: Generate Channel Weights from Text
#         # "根據這句話，我該關注音訊的哪些特徵通道？"
#         # 根據語義向量，算出 1024 個通道各自的重要性
#         # 輸入: (32, 768) -> MLP -> 輸出: (32, 1024)
#         # 數值範圍在 0.0 ~ 1.0 之間
#         channel_weights = self.se_fc(text_global) # (B, D_a)
        
#         # Expand weights to match time dimension: (B, 1, D_a)
#         # 為了跟音訊相乘，我們需要把時間維度擴展出來
#         # (32, 1024) -> (32, 1, 1024)
#         # 意思就是：這個通道的權重，對整段音訊的所有時間點都適用 (Time-invariant)
#         channel_weights = channel_weights.unsqueeze(1) 

#         # Step 3: Channel Injection (Modulation)
#         # 這是核心：Text 調整 Audio 的特徵強度
#         # Element-wise Multiplication (Hadamard product)
#         # Audio (32, 1000, 1024) * Weights (32, 1, 1024)
#         # PyTorch 會自動廣播 (Broadcasting)，把 Weights 複製 1000 份對齊時間
#         # 結果：如果不重要的特徵通道 (權重接近0)，整條時間軸上的數值都會被壓低
#         audio_modulated = audio_seq * channel_weights

#         # Step 4: Fusion & Residual
#         # 我們把調整過後的特徵加回原始特徵，防止資訊丟失
#         # 原始音訊 (audio_seq) + 調變後的音訊 (audio_modulated)
#         # 這樣做保證了：就算文字完全沒用，模型至少還能退化成原本的音訊模型，不會變爛。
#         out = self.norm(self.out_proj(audio_modulated) + audio_seq)
        
#         return out

class SemanticGuidedChannelInjection(nn.Module):
    def __init__(self, audio_dim, text_dim, reduction=4):
        super().__init__()
        self.audio_dim = audio_dim
        
        # 1. 不動 AttentivePooling，照常使用
        self.text_pool = AttentivePooling(text_dim) 
        
        # 2. 改用更穩定的 FiLM 生成器 (同時產生 Scale 和 Shift)
        # 輸出維度變兩倍: 一半給 Gamma (乘法), 一半給 Beta (加法)
        self.film_fc = nn.Sequential(
            nn.Linear(text_dim, audio_dim // reduction),
            nn.ReLU(),
            nn.Linear(audio_dim // reduction, audio_dim * 2) 
        )

        # 3. 輸出層
        self.out_proj = nn.Linear(audio_dim, audio_dim)
        self.norm = nn.LayerNorm(audio_dim)

        # === [關鍵救命丹] 初始化 (Identity Initialization) ===
        # 這是讓你的成效「至少不掉」的關鍵！
        # 強制讓初始輸出的 Gamma 為 0, Beta 為 0
        # 這樣一開始: Feature = (1+0)*Audio + 0 = Audio (完全保留原始訊號)
        with torch.no_grad():
            self.film_fc[-1].weight.fill_(0)
            self.film_fc[-1].bias.fill_(0)

    def forward(self, audio_seq, text_seq, text_mask=None):
        """
        Args:
            text_mask: (B, T) Boolean Tensor. 
                        True 代表 Padding (要遮掉), False 代表有字.
        """
        
        # === [修正 Mask] 適配 AttentivePooling ===
        # AttentivePooling 規定: "mask==0 (False) 的地方要被遮掉"
        # 我們的輸入 text_mask: "True 的地方是 Padding"
        # 所以我們必須「反轉」它 (Logical NOT):
        #   ~True (Pad)  -> False (0) -> AttentivePooling 會遮掉它 (正確!)
        #   ~False (Valid)-> True (1) -> AttentivePooling 會保留它 (正確!)
        
        if text_mask is not None:
            valid_mask = ~text_mask 
        else:
            valid_mask = None

        # 1. 提取文字語義 (現在 Mask 邏輯正確了)
        text_global = self.text_pool(text_seq, mask=valid_mask) # (B, D_t)

        # 2. 生成 FiLM 參數
        film_params = self.film_fc(text_global) # (B, D_a * 2)
        gamma, beta = torch.chunk(film_params, 2, dim=-1) # 分割成 Scale 和 Shift
        
        # 擴展維度以便廣播: (B, 1, D_a)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # 3. 執行 FiLM 調變 (Affine Transformation)
        # 公式: (1 + Gamma) * Audio + Beta
        # 這種加法式結構比單純的 Sigmoid 乘法更不容易導致梯度消失
        audio_modulated = (1.0 + gamma) * audio_seq + beta

        # 4. 殘差連接與正規化
        out = self.norm(self.out_proj(audio_modulated) + audio_seq)
        
        return out

class BaseTransformerPredictor_with_SGCI(BasePredictor):
    """
    Base class for the dual-branch transformer architecture with Semantic-Guided Channel Injection (SGCI).
    Handles feature extraction, temporal modeling, and fusion.
    """
    def __init__(self, muq_model, roberta_model,
                 audio_transformer_layers=1, audio_transformer_heads=4, audio_transformer_dim=1024,
                 common_embed_dim=768, cross_attention_heads=4, dropout_rate=0.3):
        super().__init__()
        self.muq = muq_model
        self.roberta = roberta_model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        for param in self.muq.parameters():
            param.requires_grad = False
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.muq_output_dim = 1024  # MuQ-large
        self.roberta_output_dim = 768 # RoBERTa-base

        # [修改點 1] 移除原本的 Linear Projection 到 common_embed_dim
        # 我們直接在各自的維度操作，最後再對齊，這樣可以保留原始特徵的物理意義
        # 如果維度差異太大，可以用 Linear 降維，但這裡示範直接用 SGCI

        # --- Audio Path (Temporal Modeling) ---
        self.audio_pos_encoder = PositionalEncoding(d_model=self.muq_output_dim, dropout=dropout_rate)
        
        audio_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.muq_output_dim, nhead=audio_transformer_heads,
            dim_feedforward=audio_transformer_dim * 2, dropout=dropout_rate,
            activation='relu', batch_first=True
        )
        self.audio_transformer_encoder = nn.TransformerEncoder(
            encoder_layer=audio_encoder_layer, num_layers=audio_transformer_layers
        )
        # Pooling for MI branch
        self.audio_attentive_pool = AttentivePooling(input_dim=self.muq_output_dim)

        # --- Feature Fusion (SGCI) ---
        # [修改點 2] 替換 Cross-Attention 為 SGCI
        # 注意：這裡讓 Audio 維度保持 1024，Text 維度保持 768，不強制對齊
        self.fusion_module = SemanticGuidedChannelInjection(
            audio_dim=self.muq_output_dim,  # 1024
            text_dim=self.roberta_output_dim  # 768
        )
        
        # Pooling for TA branch
        # [修改點 3] 最後的 Pooling 因為維度還是 1024 (Audio維度)，所以要改
        self.fused_attentive_pool = AttentivePooling(input_dim=self.muq_output_dim)

    def forward_features(self, wavs, texts, use_decoupled_audio_for_cross_attn=False):
        """
        Forward pass for feature extraction and fusion.
        """
        # --- Base Feature Extraction (Frozen Encoders) ---
        # Encoders are frozen in __init__. We set them to eval() mode.
        self.muq.eval()
        self.roberta.eval()

        # [修改] 強制不紀錄梯度，節省大量記憶體
        with torch.no_grad():    
            # We do not need torch.no_grad() here if requires_grad=False is set correctly. Please check if this works, this refactored code has not been verified.
            muq_output = self.muq(wavs, output_hidden_states=False)
            audio_seq_embed_raw = muq_output.last_hidden_state # (B, T_a, D_a)

            # Text Encoder
            text_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
            text_attention_mask = text_inputs['attention_mask'].to(wavs.device) # (B, T_t)
            text_inputs_on_device = {k: v.to(wavs.device) for k, v in text_inputs.items()}
            roberta_output = self.roberta(**text_inputs_on_device)
            text_seq_embed = roberta_output.last_hidden_state # (B, T_t, D_t)

        # Note: Audio padding mask is currently not utilized in the original implementation.
        audio_padding_mask = None # Transformer expects True where padded

        # --- Audio Path (Temporal Modeling) ---
        audio_seq_embed_pe = self.audio_pos_encoder(audio_seq_embed_raw)
        audio_transformed = self.audio_transformer_encoder(
            src=audio_seq_embed_pe,
            src_key_padding_mask=audio_padding_mask
        )

        # Pooling for MI branch
        pooled_audio_features = self.audio_attentive_pool(
            audio_transformed,
            mask=audio_padding_mask
        )

        
        # Determine audio input for cross-attention (Key, Value)
        if use_decoupled_audio_for_cross_attn:
            audio_input_for_fusion = audio_seq_embed_raw # Decoupled Architecture
        else:
            audio_input_for_fusion = audio_transformed # Standard Architecture

        # [修改點 4] 呼叫 Novel Fusion
        # 這裡不需要把兩者投影到 common space，直接融合

        # Text padding mask (True where PADDED)
        text_padding_mask = (text_attention_mask == 0)

        # 核心融合：Text 指導 Audio 通道權重
        # 輸出維度維持與 Audio 相同 (B, T_a, 1024)
        fused_seq = self.fusion_module(audio_input_for_fusion, text_seq_embed, text_mask=text_padding_mask)

        # Pooling for TA branch
        fused_features = self.fused_attentive_pool(
            fused_seq,
            mask=audio_padding_mask  # 注意這裡 mask 是 audio 的 mask
        )

        return pooled_audio_features, fused_features
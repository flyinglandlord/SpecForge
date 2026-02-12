# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from yunchang import EXTRACT_FUNC_DICT

from specforge.core.loss import LogSoftmaxLoss
from specforge.distributed import (
    gather_outputs_and_unpad,
    get_sp_ring_group,
    get_sp_ulysses_group,
)
from specforge.modeling.draft import Eagle3DraftModel
from specforge.utils import padding


class Eagle3Model(nn.Module):
    pass


class OnlineEagle3Model(Eagle3Model):
    """
    In sgl-spec, we implement offline/online training.
    Online training means we have the target hidden_states available during training.
    Eagle3 using test time training technique (TTT) to train the draft model.
    1. We first extract the hidden states from the target model.
    2. Then concatenate the hidden states from 3 aux layers (layer 1, layer num_layers//2, layer num_layers-4).
    3. We project the concatenated hidden states to the target hidden size. from (batch, seq_len, 3*hidden_size) to (batch, seq_len, hidden_size)
    4. We concat the projected hidden states and embedding output as the input for the draft model.
    5. finally, we run TTT to train the draft model. input size is (batch, seq_len, hidden_size * 2)
    """

    def __init__(
        self,
        draft_model: Eagle3DraftModel,
        length: int = 7,
        attention_backend="sdpa",
    ):
        """
        Args:
            target_model: the target model to extract hidden states.
            draft_model: the draft model to be trained.
            length: TTT length, it means how many turns to unroll during TTT.
        """
        super().__init__()
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend

        if self.attention_backend == "usp":
            self.extract_func = EXTRACT_FUNC_DICT["basic"]
            self.sp_ring_degree = torch.distributed.get_world_size(get_sp_ring_group())
            self.sp_ulysses_degree = torch.distributed.get_world_size(
                get_sp_ulysses_group()
            )
            self.sp_world_size = self.sp_ring_degree * self.sp_ulysses_degree
            self.sp_rank = torch.distributed.get_rank() % self.sp_world_size

    @torch.compile()
    def prepare_usp_input(self, full_input):
        shared_input = self.extract_func(
            full_input,
            rank=self.sp_rank,
            world_size=self.sp_world_size,
        ).clone()
        return shared_input

    # ================= LOSS FUNCTIONS =================
    def _compute_loss_1_topk_soft_ce(self, logits, target_probs, mask, k=16):
        # 1. 获取 Main Model (Teacher) 的 Top-K
        teacher_topk_probs, teacher_topk_indices = torch.topk(target_probs, k=k, dim=-1)
        
        # 2. 获取 Draft Model 的 Log Probabilities
        draft_log_probs_all = F.log_softmax(logits, dim=-1) # [B, S, V]
        
        # 3. Gather Draft 在 Teacher 关注的那 K 个位置的 Log Probs
        draft_log_probs_topk = draft_log_probs_all.gather(dim=-1, index=teacher_topk_indices) # [B, S, K]
        
        # 4. 计算 Partial Cross Entropy: - Sum( P_teacher_raw * log P_student )
        loss_per_token = -torch.sum(teacher_topk_probs * draft_log_probs_topk, dim=-1) # [B, S]
        
        # 5. Mask Mean
        if mask.dim() == 3: mask = mask.squeeze(-1)
        loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
        
        return loss
    
    def _compute_loss_3_ranking(self, logits, target_probs, mask):
        """
        Loss 3: Ranking
        Formula: - log( p_draft_target / p_draft_max )
        这里依然基于 Softmax (LogSoftmax)
        """
        # 使用 LogSoftmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        target_indices = target_probs.argmax(dim=-1).unsqueeze(-1)
        target_log_prob = log_probs.gather(dim=-1, index=target_indices)
        
        # 这里的 max 是在 vocab 维度取 max
        max_log_prob = log_probs.max(dim=-1, keepdim=True).values
        
        # Log 空间相减 = 概率空间相除
        loss_raw = max_log_prob - target_log_prob
        
        loss = (loss_raw * mask).sum() / (mask.sum() + 1e-8)
        return loss

    def _compute_loss_4_sigmoid_contrastive(self, logits, target_probs, mask):
        """
        Loss 4: Sigmoid Contrastive (One-vs-Rest)
        Formula: -log(p_target) - mean(log(1 - p_others))
        注意: 只有这个 Loss 应该视 Logits 为 Sigmoid 输入
        """
        # 这里可以使用 binary_cross_entropy_with_logits 或者手动 sigmoid
        # 为了对应你给的公式逻辑，手动 sigmoid 更清晰
        probs = torch.sigmoid(logits)
        target_indices = target_probs.argmax(dim=-1).unsqueeze(-1)
        
        # Target: maximize log(p)
        p_target = probs.gather(dim=-1, index=target_indices)
        term_target = -torch.log(p_target + 1e-8)
        
        # Others: maximize log(1-p) -> minimize p
        # sum(log(1-p_all)) - log(1-p_target)
        sum_log_inv = torch.log(1 - probs + 1e-8).sum(dim=-1, keepdim=True)
        log_inv_target = torch.log(1 - p_target + 1e-8)
        
        # 剩下 V-1 个词的平均值
        term_others = - (sum_log_inv - log_inv_target) / (logits.size(-1) - 1)
        
        loss = ((term_target + term_others) * mask).sum() / (mask.sum() + 1e-8)
        return loss

    def _compute_loss_top_p_renorm_ce(self, logits, target_probs, mask, top_p=0.9, max_k_search=256):
        """
        Component 1: Top-P Renormalized Soft Cross Entropy
        只关注 Teacher 认为靠谱的 Token，忽略尾部噪声。
        """
        B, S, V = target_probs.shape
    
        # 1. 只取 Top-K 值和索引，避免对 128k 词表进行全排序
        # max_k_search 设为 128 或 256 足够覆盖 0.9 的概率质量
        # memory usage: O(B*S*K) vs O(B*S*V)
        top_k_probs, top_k_indices = torch.topk(target_probs, k=min(max_k_search, V), dim=-1) # [B, S, K]
        
        # 2. 在 K 的维度上计算累积概率
        cumulative_probs = torch.cumsum(top_k_probs, dim=-1)
        
        # 3. 生成 Mask (逻辑同原代码，只是在 K 维度操作)
        sorted_indices_to_remove = cumulative_probs > top_p
        # shift right to keep the first token that crosses the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 4. 重新归一化 (只在这个 Top-K 的子集上操作)
        # 既然我们已经放弃了 K 以外的词，逻辑上：
        # valid_probs = top_k_probs * (~remove_mask)
        top_k_probs_clean = top_k_probs.masked_fill(sorted_indices_to_remove, 0.0)
        
        # 计算归一化因子
        normalization_factor = top_k_probs_clean.sum(dim=-1, keepdim=True) + 1e-8
        target_probs_renorm_small = top_k_probs_clean / normalization_factor # [B, S, K]
        
        # 5. 计算 Loss: 只需取出 Logits 对应的 Top-K 位置
        # 这样避免了构建一个巨大的 Full Vocab Mask
        log_probs = F.log_softmax(logits, dim=-1) # [B, S, V]
        
        # Gather draft log_probs at top_k indices
        # index 维度必须和 output 维度匹配，所以用 top_k_indices
        log_probs_gathered = log_probs.gather(dim=-1, index=top_k_indices) # [B, S, K]
        
        # 计算 Cross Entropy: - sum( P_small * log Q_small )
        loss_per_token = -torch.sum(target_probs_renorm_small * log_probs_gathered, dim=-1) # [B, S]
        
        # 6. Mask Mean
        if mask.dim() == 3: mask = mask.squeeze(-1)
        loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)
        
        return loss

    # ================ DYNAMIC LENGTH HANDLING =================
    def fix_target_p_dynamic(
        self,
        target_p: torch.Tensor,
        dynamic_lengths: torch.Tensor,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理动态训练逻辑：
        不再修改 target_p 的数值（不插入 IDK），而是根据 dynamic_lengths 生成 Mask。
        
        Args:
            target_p: [B, S, V]
            dynamic_lengths: [B, S] 截断位置
            idx: 当前 TTT 的 step 索引
        Returns:
            target_p: 原封不动返回
            dynamic_mask: [B, S, 1] 只有在 idx < dynamic_lengths 的位置为 1
        """
        # 扩展维度以匹配
        idx_tensor = torch.full_like(dynamic_lengths, idx)
        
        # Logic: 只有在截断点之前的位置参与训练
        # idx < length -> Train
        # idx >= length -> Ignore
        mask = (idx_tensor < dynamic_lengths).unsqueeze(-1).int()
        
        return target_p, mask

    def fix_target_p_with_idk(
        self,
        target_p: torch.Tensor,          # [B, S, V] 原始主模型输出
        dynamic_lengths: torch.Tensor,   # [B, S] 截断位置
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        修改目标概率分布以训练 <IDK>。
        Logic:
        1. Match (idx < len): Target_IDK = 1.0 - P_Main(Top1). 
           含义: 即使正确，如果不自信，也要提高 IDK 概率。
        2. Mismatch (idx == len): Target_IDK = 1.0. 
           含义: 此处必须截断。
        3. Ignore (idx > len): Mask out.
        """
        B, S, V = target_p.shape
        device = target_p.device
        dtype = target_p.dtype
        idk_index = self.draft_model.idk_index.item() if self.draft_model.idk_index is not None else None

        # 如果 IDK token 是词表外的新 token，拼接一列
        need_extra_idk = (idk_index is not None) and (idk_index >= V)
        if need_extra_idk:
            idk_col = torch.zeros((B, S, 1), device=device, dtype=dtype)
            # clone 确保不修改原始引用 (虽然下面还是会修改)
            target_p = torch.cat([target_p, idk_col], dim=-1)
        else:
            target_p = target_p.clone()

        idx_tensor = torch.full_like(dynamic_lengths, idx)

        # Masks
        lt_mask = idx_tensor < dynamic_lengths     # Match: Soft IDK
        eq_mask = idx_tensor == dynamic_lengths    # Mismatch: Hard IDK
        gt_mask = idx_tensor > dynamic_lengths     # Ignore

        # --- Step 1: 处理 Match (Soft Label) ---
        if lt_mask.any():
            # 获取主模型当前的 Top-1 置信度
            # 注意：取 max 时要避开 IDK 列 (如果 IDK 在最后)
            vocab_probs = target_p[..., :idk_index] if need_extra_idk else target_p
            top_vals, _ = vocab_probs.max(dim=-1)   # [B, S]
            top_vals = top_vals.detach() # 梯度截断，Target 是常数

            # 代理目标: IDK = 1 - Confidence
            soft_idk_target = 1.0 - top_vals 
            
            # 填入 IDK 位置
            target_p[lt_mask, idk_index] = soft_idk_target[lt_mask]
        
        # --- Step 2: 处理 Mismatch (Hard Label) ---
        if eq_mask.any():
            # 在截断点，强制 IDK 为 1.0
            target_p[eq_mask, idk_index] = 1.0
            # 如果希望此时 CE Loss 不受干扰，target_non_idk 不变即可
            # BCE Loss 负责拉高 IDK logit，CE Loss 负责让它即便在截断处也预测正确的 token
            # 这样有利于 tree structure 的完整性

        # --- Step 3: 生成参与 Loss 计算的 Mask ---
        # 只有 <= dynamic_lengths 的位置参与训练
        dynamic_position_mask = (~gt_mask).unsqueeze(-1).int()

        return target_p, dynamic_position_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        """
        Online eagle model trainer with detailed logging.

        Returns:
            plosses: list of total losses per step (for backward)
            vlosses: empty list (legacy compatibility)
            acces: list of accuracies per step
            training_details: Dict containing detailed metrics (bce_loss, rank_loss, pdiff, etc.)
        """
        # Step 1: handle vocab size
        target_p_padded, position_mask = _compute_target_p_padded(
            target=target,
            t2d=self.draft_model.t2d,
            loss_mask=loss_mask,
            length=self.length,
        )
        del target
        dynamic_lengths = kwargs.get("dynamic_lengths", None)
        use_dynamic_training = dynamic_lengths is not None

        # basic info
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # Step 2: project hidden states
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        # Step 3: process kv cache, position ids
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # Step 4: handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if self.attention_backend in ("sdpa", "usp"):
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        # Step 5: run TTT
        plosses = []  # Total loss (variables with grad)
        vlosses = []
        acces = []

        # --- New: Detailed Metrics Lists ---
        det_bce_losses = []
        det_rank_losses = []
        det_pdiffs = []

        global_input_ids = input_ids
        
        # Init Cache based on backend
        if self.attention_backend in ["sdpa", "fa"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        elif self.attention_backend == "usp":
            cache_hidden = [[], []]
            past_key_values = None
            hidden_states = self.prepare_usp_input(hidden_states)
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        for idx in range(self.length):
            target_p = target_p_padded[:, idx : idx + seq_length, :]

            if use_dynamic_training:
                # Generate Mask based on dynamic lengths
                _, dynamic_mask = self.fix_target_p_dynamic(
                    target_p=target_p,
                    dynamic_lengths=dynamic_lengths,
                    idx=idx,
                )
                current_position_mask = position_mask * dynamic_mask
            else:
                current_position_mask = position_mask

            if self.attention_backend == "usp":
                input_ids = self.prepare_usp_input(global_input_ids)
            else:
                input_ids = global_input_ids

            is_last = idx == self.length - 1

            # Step 5.1: embed
            inputs_embeds = self.draft_model.embed_input_ids(input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            # Step 5.2: backbone
            hidden_states_out = self.draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            hidden_states = hidden_states_out

            # Step 5.4: get logits
            logits = self.draft_model.compute_logits(hidden_states)
            logits = gather_outputs_and_unpad(logits, gather_dim=1)

            if use_dynamic_training:
                loss_bce = self._compute_loss_1_topk_soft_ce(logits, target_p, current_position_mask, k=16)
                loss_rank = self._compute_loss_3_ranking(logits, target_p, current_position_mask)
                
                # 3. 总 Loss (用于反向传播)
                loss = loss_bce + loss_rank
                
                with torch.no_grad():
                    det_bce_losses.append(loss_bce.clone().detach())
                    det_rank_losses.append(loss_rank.clone().detach())
            else:
                # Standard Legacy Training
                loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)

            plosses.append(loss)

            # Step 5.5: record metrics
            with torch.no_grad():
                acces.append(
                    _compute_metric_acc(
                        logits=logits,
                        target_p=target_p,
                        position_mask=current_position_mask,
                        loss_mask=loss_mask,
                    )
                )
                det_pdiffs.append(
                    _compute_top_prob_diff(
                        logits=logits,
                        target_p=target_p,
                        valid_mask=current_position_mask
                    )
                )

            if not is_last:
                global_input_ids = padding(global_input_ids, left=False)
                position_mask = padding(position_mask, left=False)
                loss_mask = padding(loss_mask, left=False)

        # 构造返回字典
        training_details = {}
        if use_dynamic_training:
            if len(det_bce_losses) > 0:
                training_details["detail_bce_loss"] = det_bce_losses
            if len(det_rank_losses) > 0:
                training_details["detail_rank_loss"] = det_rank_losses
            training_details["dynamic_lengths"] = dynamic_lengths
        training_details["detail_pdiff"] = det_pdiffs
        
        return plosses, vlosses, acces, training_details


class QwenVLOnlineEagle3Model(Eagle3Model):
    """
    In sgl-spec, we implement offline/online training.
    Online training means we have the target hidden_states available during training.
    Eagle3 using test time training technique (TTT) to train the draft model.
    1. We first extract the hidden states from the target model.
    2. Then concatenate the hidden states from 3 aux layers (layer 1, layer num_layers//2, layer num_layers-4).
    3. We project the concatenated hidden states to the target hidden size. from (batch, seq_len, 3*hidden_size) to (batch, seq_len, hidden_size)
    4. We concat the projected hidden states and embedding output as the input for the draft model.
    5. finally, we run TTT to train the draft model. input size is (batch, seq_len, hidden_size * 2)
    """

    def __init__(
        self,
        target_model,
        draft_model: Eagle3DraftModel,
        processor,
        length: int = 7,
        attention_backend: str = "sdpa",
    ):
        """
        Args:
            target_model: the target model to extract hidden states.
            draft_model: the draft model to be trained.
            length: TTT length, it means how many turns to unroll during TTT.
        """
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.processor = processor
        self.length = length
        self.attention_backend = attention_backend

    @torch.no_grad()
    def _prepare_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L692
        Extract the hidden states from the target model outputs.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            device: the device to run the target model, if None, use the input_ids device
            pixel_values: image pixel values, used for VLM models
            image_grid_thw: image grid thw, used for VLM models

        Returns:
            hidden_states: (batch, seq_len, 3*hidden_size)
            target: (batch, seq_len, vocab_size)
            loss_mask: (batch, seq_len)
            input_ids: (batch, seq_len)
        """

        if device is None:
            device = input_ids.device

        # run the target model to get the hidden states
        outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=False,
        )

        # extract the aux hidden states
        # output_hidden_states = True will return the embedding output as well
        # so we have an offset of 1
        num_hidden_states = len(outputs.hidden_states)
        offset = 1
        num_layers = num_hidden_states - 1

        # Eagle3 uses 3 aux layers from layer 1, num_layers//2, num_layers-4
        low_aux_layer = 1 + offset
        mid_aux_layer = num_layers // 2 - 1 + offset
        last_aux_layer = num_layers - 4 + offset

        hidden_states0 = outputs.hidden_states[low_aux_layer]
        hidden_states1 = outputs.hidden_states[mid_aux_layer]
        hidden_states2 = outputs.hidden_states[last_aux_layer]

        hidden_states = torch.cat(
            (hidden_states0, hidden_states1, hidden_states2), dim=-1
        )

        # apply pading
        target = outputs.logits
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)

        if target is not None:
            target = target.to(device)
            loss_mask = loss_mask[..., None]
            loss_mask = loss_mask.to(device)

        return hidden_states, target, loss_mask, input_ids

    @torch.no_grad()
    def _get_input_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # get input embeding with image
        # inputs_embeds = self.target_model.model.get_input_embeddings()(input_ids)
        inputs_embeds = self.draft_model.embed_input_ids(input_ids)
        image_embeds = self.target_model.model.get_image_features(
            pixel_values, image_grid_thw
        )
        image_embeds = torch.cat(image_embeds, dim=0)
        n_image_tokens = (
            input_ids == self.target_model.model.config.image_token_id
        ).sum()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == self.target_model.model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Online eagle model trainer, modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L711

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            past_key_values: We dont use this past_key_values in eagle3, but keep it for compatibility. We control kvcache by cache_hidden.
            position_ids: (batch, seq_len)
            pixel_values: batch image pixel values, used for VLM models
            image_grid_thw: (batch, 3), image grid thw, used for VLM models
        """
        # Step 0: prepare data with the target model
        hidden_states, target, loss_mask, input_ids = self._prepare_data(
            input_ids, attention_mask, loss_mask, pixel_values, image_grid_thw
        )

        # Step 1: handle vocab size
        target_p_padded, position_mask = _compute_target_p_padded(
            target=target,
            t2d=self.draft_model.t2d,
            loss_mask=loss_mask,
            length=self.length,
        )
        del target

        # basic info
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # Step 2: project the concatenated hidden states to the target hidden size
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        # Step 3: process kv cache, position ids and position ids
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                attention_mask_tensor = (
                    attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                )
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            position_ids, rope_deltas = self.target_model.model.get_rope_index(
                input_ids,
                image_grid_thw,
                None,
                second_per_grid_ts=None,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            position_ids = position_ids

        # Step 4: handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        # Step 5: run TTT
        plosses = []
        vlosses = []
        acces = []
        if self.attention_backend in ["sdpa", "fa"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        for idx in range(self.length):
            target_p = target_p_padded[:, idx : idx + seq_length, :].contiguous()
            is_last = idx == self.length - 1

            # Step 5.1: embed the input ids
            # inputs_embeds = self._get_input_embeds(input_ids, pixel_values, image_grid_thw)
            inputs_embeds = self.draft_model.embed_input_ids(input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            # Step 5.2: run the draft model backbone
            hidden_states_out = self.draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # update hidden states for next step
            hidden_states = hidden_states_out

            # Step 5.4: get logits
            logits = self.draft_model.compute_logits(hidden_states)

            # Step 5.5: record metrics first as we in-place modify logits
            with torch.no_grad():
                acces.append(
                    _compute_metric_acc(
                        logits=logits,
                        target_p=target_p,
                        position_mask=position_mask,
                        loss_mask=loss_mask,
                    )
                )

            # Step 5.6: calculate loss, in-place modifies logits!
            loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)
            plosses.append(loss)

            if not is_last:
                # Step 5.7: we need to update the loss mask
                input_ids = padding(input_ids, left=False)
                position_mask = padding(position_mask, left=False)
                loss_mask = padding(loss_mask, left=False)
                # Flex attention mask shirnking is handled inside attention module
        return plosses, vlosses, acces


def _compute_target_p_padded(target, t2d, loss_mask, length):
    with torch.no_grad():
        target_p, position_mask = _compute_target_p(
            target=target,
            t2d=t2d,
            loss_mask=loss_mask,
        )

        assert len(target_p.shape) == 3
        target_p_padded = F.pad(
            target_p,
            pad=(0, 0, 0, length),
            mode="constant",
            # For bitwise equality with previous code
            value=1 / target_p.shape[-1],
        )

        return target_p_padded, position_mask


@torch.compile(dynamic=None)
def _compute_target_p(target, t2d, loss_mask):
    target_head = target
    target_max_token = target_head.argmax(-1)
    target_mask = t2d[target_max_token]
    target_mask = target_mask[..., None].int()
    position_mask = target_mask * loss_mask
    target_head = target_head[..., t2d]
    target_head = target_head.float()
    target_p = nn.Softmax(dim=2)(target_head)
    target_p = target_p.detach()
    return target_p, position_mask


@torch.compile(dynamic=None)
def _compute_metric_acc(logits, target_p, position_mask, loss_mask):
    return (
        (logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)
    ).sum() / loss_mask.sum().clamp_min(1e-6)


def _compute_top_prob_diff(
    logits: torch.Tensor, 
    target_p: torch.Tensor, 
    valid_mask: torch.Tensor
) -> torch.Tensor:
    """
    计算 Draft Model 和 Main Model 在 Top-1 概率上的平均绝对误差 (MAE)。
    
    Args:
        logits: Draft model outputs [Batch, Seq, Vocab]
        target_p: Main model probability distribution [Batch, Seq, Vocab]
        valid_mask: Combined mask (padding + dynamic cutting) [Batch, Seq]
    """
    with torch.no_grad():
        # 1. Draft Top1 Prob (Standardize with Softmax for comparison)
        draft_probs = F.softmax(logits, dim=-1)
        draft_top1 = draft_probs.max(dim=-1).values # [B, S]
        
        # 2. Main Top1 Prob (Target is already probability)
        main_top1 = target_p.max(dim=-1).values # [B, S]
        
        # 3. Calculate |P_draft - P_main|
        # Ensure mask is [B, S]
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.squeeze(-1)
            
        diff = torch.abs(draft_top1 - main_top1) * valid_mask 
        
        # 4. Average
        avg_diff = diff.sum() / (valid_mask.sum() + 1e-8)
        
        return avg_diff

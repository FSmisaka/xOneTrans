"""Independent OneTrans implementation for sequence-representation diagnostics.

Re-implements the UniRank OneTrans structure (model_zoo/OneTrans.py, upstream
commit 90bb99a) with a diagnostic `return_sequence_states` option: the model
returns the backbone input and the post-update sequence tokens of every layer,
together with the sequence validity mask. Structural parameters and the
dual-stream (sequence tokens / non-sequence tokens) per-layer interaction are
kept identical to the reference; `reduction_ratio` is fixed to 0.0 for all
diagnostics so sequence positions are stable across layers.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


ACTIVATIONS = {"ReLU": nn.ReLU, "SiLU": nn.SiLU, "GeLU": nn.GELU, "Tanh": nn.Tanh}


def get_activation(name: str) -> nn.Module:
    return ACTIVATIONS[name]()


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_units, hidden_activations="ReLU",
                 output_dim=None, layer_norm=False, dropout_rates=0.0):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_units) + ([output_dim] if output_dim else [])
        n_hidden = len(dims) - 1
        for i in range(n_hidden):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = output_dim is not None and i == n_hidden - 1
            if not is_last:
                layers.append(get_activation(hidden_activations))
                if layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                if dropout_rates > 0:
                    layers.append(nn.Dropout(dropout_rates))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AutoSplitTokenizer(nn.Module):
    """`Auto` unified tokenizer: one MLP per generated NS token."""

    def __init__(self, input_dim, token_dim, num_tokens, activation="SiLU", layer_norm=True):
        super().__init__()
        self.token_mlps = nn.ModuleList([
            MLPBlock(input_dim, [token_dim], hidden_activations=activation, layer_norm=layer_norm)
            for _ in range(num_tokens)
        ])

    def forward(self, feature_embeddings):
        if feature_embeddings.dim() > 2:
            feature_embeddings = torch.flatten(feature_embeddings, start_dim=1)
        return torch.stack([mlp(feature_embeddings) for mlp in self.token_mlps], dim=1)


class ScaledDotProductAttention(nn.Module):
    """SoftMax scaled dot-product attention via SDPA (FlashAttention eligible)."""

    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, q, k, v, is_causal=True):
        return F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_rate if self.training else 0.0,
            is_causal=is_causal,
        )


class MixedMHA(nn.Module):
    """Mixed multi-head attention: shared QKV projections for sequence tokens,
    token-specific projections for NS tokens; causal attention over the
    concatenated [S; NS] stream (identical to the reference implementation)."""

    def __init__(self, input_dim, num_heads, num_ns_token):
        super().__init__()
        assert input_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.W_q_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v_s = nn.Linear(input_dim, input_dim, bias=False)
        self.W_q_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_k_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_v_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, input_dim))
        self.W_o = nn.Linear(input_dim, input_dim, bias=False)
        self.dot_attention = ScaledDotProductAttention()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q_ns)
        nn.init.xavier_uniform_(self.W_k_ns)
        nn.init.xavier_uniform_(self.W_v_ns)

    def forward(self, s_tokens, ps_tokens, ns_tokens, kv_mask=None, q_mask=None):
        B, Ls, D = s_tokens.shape
        _, qLs, _ = ps_tokens.shape
        _, Lns, _ = ns_tokens.shape
        if kv_mask is not None:
            s_tokens = s_tokens * kv_mask.unsqueeze(-1).to(s_tokens.dtype)
        q_s, k_s, v_s = self.W_q_s(s_tokens), self.W_k_s(s_tokens), self.W_v_s(s_tokens)
        q_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_q_ns)
        k_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_k_ns)
        v_ns = torch.einsum("bld,ldh->blh", ns_tokens, self.W_v_ns)
        q = torch.cat([q_s, q_ns], dim=1)
        k = torch.cat([k_s, k_ns], dim=1)
        v = torch.cat([v_s, v_ns], dim=1)
        L = Ls + Lns
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        output = self.dot_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.W_o(output)
        ps_out = output[:, Ls - qLs:Ls, :]
        ns_out = output[:, Ls:, :]
        if q_mask is not None:
            ps_out = ps_out * q_mask.unsqueeze(-1).to(ps_out.dtype)
        return ps_out, ns_out


class MixedFFN(nn.Module):
    """Shared FFN for sequence tokens, token-specific FFN for NS tokens."""

    def __init__(self, input_dim, num_ns_token, dnn_activations="ReLU", expansion_factor=4):
        super().__init__()
        hidden_dim = input_dim * expansion_factor
        self.activation = get_activation(dnn_activations)
        self.ffn_s = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(dnn_activations),
            nn.Linear(hidden_dim, input_dim),
        )
        self.W1_ns = nn.Parameter(torch.empty(num_ns_token, input_dim, hidden_dim))
        self.b1_ns = nn.Parameter(torch.zeros(num_ns_token, hidden_dim))
        self.W2_ns = nn.Parameter(torch.empty(num_ns_token, hidden_dim, input_dim))
        self.b2_ns = nn.Parameter(torch.zeros(num_ns_token, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1_ns)
        nn.init.xavier_uniform_(self.W2_ns)
        nn.init.zeros_(self.b1_ns)
        nn.init.zeros_(self.b2_ns)

    def forward(self, s_tokens, ns_tokens, mask=None):
        s_out = self.ffn_s(s_tokens)
        ns_hidden = torch.einsum("bld,ldh->blh", ns_tokens, self.W1_ns) + self.b1_ns.unsqueeze(0)
        ns_hidden = self.activation(ns_hidden)
        ns_out = torch.einsum("blh,lhd->bld", ns_hidden, self.W2_ns) + self.b2_ns.unsqueeze(0)
        if mask is not None:
            s_out = s_out * mask.unsqueeze(-1).to(s_out.dtype)
        return s_out, ns_out


class ResidualRepresentationGate(nn.Module):
    """Task-aware residual representation gate (SPEC2 section 3).

    Z_r = W2(GELU(W1(LN(H)))) applied per token; gate G = sigmoid(MLP([h_p; E_t]))
    with h_p the masked mean-pooled sequence summary and E_t the (projected)
    target item id embedding; output Z_out = H + G[:, None, :] * Z_r, shared
    across the history dimension per (sample, dim). Modes:
    - "learned": full task-aware gate (spec 2.1 / 2.2 Learnable Gate)
    - "fixed":   G = alpha, a fixed scalar (spec 2.2 Fixed Gate)
    - "ungated": G = 1 (spec 2.2 Ungated Residual)
    Padding positions are zeroed following the backbone mask convention."""

    def __init__(self, token_dim, item_emb_dim, residual_hidden_dim, gate_hidden_dim,
                 mode="learned", dropout=0.0, init_bias=0.0, fixed_alpha=0.5):
        super().__init__()
        self.mode = mode
        self.fixed_alpha = fixed_alpha
        self.norm = nn.LayerNorm(token_dim)
        self.w1 = nn.Linear(token_dim, residual_hidden_dim)
        self.w2 = nn.Linear(residual_hidden_dim, token_dim)
        if mode == "learned":
            self.item_proj = nn.Linear(item_emb_dim, token_dim)
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * token_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden_dim, token_dim),
            )
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.constant_(self.gate_mlp[-1].bias, init_bias)
        self.last_zr: Optional[torch.Tensor] = None
        self.last_g: Optional[torch.Tensor] = None

    def forward(self, h, item_emb, mask, gate_override=None):
        zr = self.w2(F.gelu(self.w1(self.norm(h))))
        zr = zr * mask.unsqueeze(-1).to(zr.dtype)
        if self.mode == "learned":
            h_p = (h * mask.unsqueeze(-1).to(h.dtype)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0).to(h.dtype)
            g = torch.sigmoid(self.gate_mlp(torch.cat([h_p, self.item_proj(item_emb)], dim=-1)))
        elif self.mode == "fixed":
            g = torch.full((h.shape[0], h.shape[2]), self.fixed_alpha, device=h.device, dtype=h.dtype)
        else:
            g = torch.ones(h.shape[0], h.shape[2], device=h.device, dtype=h.dtype)
        if gate_override == "zero":
            g = torch.zeros_like(g)
        elif gate_override == "one":
            g = torch.ones_like(g)
        self.last_zr = zr
        self.last_g = g
        return h + g.unsqueeze(1) * zr


class OneTransBlock(nn.Module):
    """Pre-norm dual-stream backbone. With reduction_ratio r, each layer drops
    the first r * L sequence query tokens (NS tokens are never pruned)."""

    def __init__(self, input_dim, num_heads, num_layers, num_ns_token,
                 dnn_activations="ReLU", expansion_factor=4, reduction_ratio=0.0,
                 net_dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.reduction_ratio = float(reduction_ratio)
        if not 0.0 <= self.reduction_ratio < 1.0:
            raise ValueError(f"reduction_ratio must be in [0, 1), got {self.reduction_ratio}")
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
        self.mha_layers = nn.ModuleList([
            MixedMHA(input_dim, num_heads, num_ns_token) for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            MixedFFN(input_dim, num_ns_token, dnn_activations, expansion_factor) for _ in range(num_layers)
        ])

    def _forward_layer(self, s_tokens, ps_tokens, ns_tokens, mask, q_mask, layer_idx):
        sequence_start = int(ps_tokens.size(1) * self.reduction_ratio)
        if sequence_start > 0:
            ps_tokens = ps_tokens[:, sequence_start:, :]
            if q_mask is not None:
                q_mask = q_mask[:, sequence_start:]
        norm_s = self.norm1_layers[layer_idx](s_tokens)
        norm_ps = self.norm1_layers[layer_idx](ps_tokens)
        norm_ns = self.norm1_layers[layer_idx](ns_tokens)
        delta_ps, delta_ns = self.mha_layers[layer_idx](norm_s, norm_ps, norm_ns,
                                                        kv_mask=mask, q_mask=q_mask)
        ps_tokens = ps_tokens + delta_ps
        ns_tokens = ns_tokens + delta_ns
        norm_ps = self.norm2_layers[layer_idx](ps_tokens)
        norm_ns = self.norm2_layers[layer_idx](ns_tokens)
        delta_ps, delta_ns = self.ffn_layers[layer_idx](norm_ps, norm_ns, q_mask)
        ps_tokens = ps_tokens + delta_ps
        ns_tokens = ns_tokens + delta_ns
        return ps_tokens, ns_tokens, q_mask

    def forward_prefix(self, s_tokens, ns_tokens, mask=None, end_layer=None):
        """Run blocks [0, end_layer) and return the S/NS streams entering the next layer."""
        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).to(s_tokens.dtype)
            q_mask = mask
        else:
            q_mask = None
        ps_tokens = s_tokens
        end_layer = self.num_layers - 1 if end_layer is None else end_layer
        for i in range(end_layer):
            ps_tokens, ns_tokens, q_mask = self._forward_layer(
                s_tokens, ps_tokens, ns_tokens, mask, q_mask, i)
            s_tokens = ps_tokens
            mask = q_mask
        return ps_tokens, ns_tokens, q_mask

    def forward_one_layer(self, s_tokens, ns_tokens, mask=None, layer_idx=None):
        """Run one shared block; used by Exp4 after inserting a spectral transform."""
        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).to(s_tokens.dtype)
            q_mask = mask
        else:
            q_mask = None
        ps_tokens = s_tokens
        layer_idx = self.num_layers - 1 if layer_idx is None else layer_idx
        ps_tokens, ns_tokens, q_mask = self._forward_layer(
            s_tokens, ps_tokens, ns_tokens, mask, q_mask, layer_idx)
        return ps_tokens, ns_tokens, q_mask

    def forward(self, s_tokens, ns_tokens, mask=None, return_states=False, s_transform=None):
        if mask is not None:
            s_tokens = s_tokens * mask.unsqueeze(-1).to(s_tokens.dtype)
            q_mask = mask
        else:
            q_mask = None
        ps_tokens = s_tokens
        states: List[torch.Tensor] = []
        if return_states:
            states.append(ps_tokens)
        if s_transform is not None and self.num_layers > 0:
            ps_tokens = s_transform(ps_tokens)
            s_tokens = ps_tokens
            if return_states:
                states.append(ps_tokens)
        for i in range(self.num_layers):
            ps_tokens, ns_tokens, q_mask = self._forward_layer(
                s_tokens, ps_tokens, ns_tokens, mask, q_mask, i)
            s_tokens = ps_tokens
            mask = q_mask
            if return_states:
                states.append(ps_tokens)
        if return_states:
            return ns_tokens, states, ps_tokens
        return ns_tokens, ps_tokens


class OneTransDiagnostic(nn.Module):
    """OneTrans with per-field input masking and optional per-layer sequence
    state extraction. The variable-field embeddings are zeroed (not removed)
    when a field is disabled, keeping tokenizer slots and parameter count
    fixed. The target item id is always active."""

    FIELD_ORDER = ["user", "context", "item", "item_id"]

    def __init__(self, vocab_sizes: Dict[str, int], model_cfg: Dict):
        super().__init__()
        emb = model_cfg["embedding_dim"]
        token_dim = model_cfg["token_dim"]
        self.embedding_dim = emb
        self.token_dim = token_dim
        self.max_history_len = model_cfg["max_history_len"]

        self.item_embedding = nn.Embedding(vocab_sizes["item_id"], emb, padding_idx=0)
        self.action_embedding = nn.Embedding(vocab_sizes["action"], emb, padding_idx=0)
        self.variable_fields = list(vocab_sizes["variable_fields"].keys())
        self.field_embeddings = nn.ModuleDict({
            f: nn.Embedding(vocab_sizes["variable_fields"][f], emb, padding_idx=0)
            for f in self.variable_fields
        })
        self.num_fields = len(self.variable_fields) + 1  # + target item id

        item_info_dim = 2 * emb  # history item id + action
        self.item_token_proj = (nn.Linear(item_info_dim, token_dim)
                                if item_info_dim != token_dim else nn.Identity())

        tokenizer_input_dim = self.num_fields * emb
        assert str(model_cfg["tokenizer_type"]).lower() == "auto"
        self.unified_tokenizer = AutoSplitTokenizer(
            tokenizer_input_dim, token_dim, model_cfg["num_ns_token"],
            activation=model_cfg.get("tokenizer_activation", "SiLU"),
        )
        self.num_ns_token = model_cfg["num_ns_token"]

        self.backbone = OneTransBlock(
            input_dim=token_dim,
            num_heads=model_cfg["num_heads"],
            num_layers=model_cfg["num_layers"],
            num_ns_token=self.num_ns_token,
            dnn_activations=model_cfg["dnn_activations"],
            expansion_factor=model_cfg["expansion_factor"],
            reduction_ratio=model_cfg["reduction_ratio"],
            net_dropout=model_cfg.get("net_dropout", 0.0),
        )
        self.tower = MLPBlock(
            token_dim * self.num_ns_token, model_cfg["tower_hidden_units"],
            hidden_activations=model_cfg["dnn_activations"], output_dim=1,
            dropout_rates=model_cfg.get("net_dropout", 0.0),
        )

        # Constructed last so baseline parameter init (RNG draw order) is
        # untouched; when disabled the module does not exist and the forward
        # path is bitwise identical to the original model.
        self.residual_gate = None
        self.gate_mode = None
        if model_cfg.get("use_residual_gate", False):
            gate_cfg = model_cfg.get("residual_gate", {})
            self.gate_mode = gate_cfg.get("mode", "learned")
            self.residual_gate = ResidualRepresentationGate(
                token_dim=token_dim,
                item_emb_dim=emb,
                residual_hidden_dim=gate_cfg.get("residual_hidden_dim", 2 * token_dim),
                gate_hidden_dim=gate_cfg.get("gate_hidden_dim", 2 * token_dim),
                mode=self.gate_mode,
                dropout=gate_cfg.get("gate_dropout", 0.0),
                init_bias=gate_cfg.get("gate_init_bias", 0.0),
                fixed_alpha=gate_cfg.get("fixed_gate_alpha", 0.5),
            )

    def param_groups(self, dense_lr: float, sparse_lr: float):
        sparse_ids = {id(p) for emb in [self.item_embedding, self.action_embedding, *self.field_embeddings.values()]
                      for p in emb.parameters()}
        dense, sparse = [], []
        for p in self.parameters():
            (sparse if id(p) in sparse_ids else dense).append(p)
        return [{"params": dense, "lr": dense_lr}, {"params": sparse, "lr": sparse_lr}]

    def encode_inputs(self, batch: Dict[str, torch.Tensor],
                      field_mask: Optional[torch.Tensor] = None):
        hist_item = batch["hist_item"]
        hist_action = batch["hist_action"]
        hist_mask = batch["hist_mask"].to(torch.float32)

        seq_emb = torch.cat([self.item_embedding(hist_item),
                             self.action_embedding(hist_action)], dim=-1)
        s_tokens = self.item_token_proj(seq_emb)
        s_tokens = s_tokens * hist_mask.unsqueeze(-1)

        if field_mask is None:
            field_mask = torch.ones(self.num_fields, device=s_tokens.device)
        field_mask = field_mask.to(s_tokens.device, s_tokens.dtype)
        pieces = [self.field_embeddings[f](batch[f]) for f in self.variable_fields]
        item_id_piece = self.item_embedding(batch["item_id"])
        field_emb = torch.cat(pieces + [item_id_piece], dim=-1)
        feature_embeddings = field_emb * field_mask.repeat_interleave(self.embedding_dim).unsqueeze(0)

        ns_tokens = self.unified_tokenizer(feature_embeddings)
        return s_tokens, ns_tokens, hist_mask

    def encode_prefix(self, batch: Dict[str, torch.Tensor],
                      field_mask: Optional[torch.Tensor] = None, end_layer: Optional[int] = None):
        """Return S/NS streams entering the block after `end_layer` blocks."""
        if self.residual_gate is not None:
            raise ValueError("Exp4 prefix/one-layer API does not support the residual gate")
        s_tokens, ns_tokens, hist_mask = self.encode_inputs(batch, field_mask)
        end_layer = self.backbone.num_layers - 1 if end_layer is None else end_layer
        s_pre, ns_pre, mask = self.backbone.forward_prefix(
            s_tokens, ns_tokens, hist_mask, end_layer=end_layer)
        return s_pre, ns_pre, mask

    def final_layer(self, s_tokens: torch.Tensor, ns_tokens: torch.Tensor,
                    mask: Optional[torch.Tensor], layer_idx: Optional[int] = None):
        """Run one shared block plus the original CTR tower."""
        s_final, ns_final, mask_out = self.backbone.forward_one_layer(
            s_tokens, ns_tokens, mask, layer_idx=layer_idx)
        readout = ns_final.flatten(start_dim=1)
        logit = self.tower(readout).squeeze(-1)
        logit = logit + 0.0 * s_final.sum()
        return {"logit": logit, "pred": torch.sigmoid(logit), "readout": readout,
                "s_final": s_final, "ns_final": ns_final, "hist_mask": mask_out}

    def forward(self, batch: Dict[str, torch.Tensor], field_mask: Optional[torch.Tensor] = None,
                return_sequence_states: bool = False, gate_override: Optional[str] = None,
                return_gate: bool = False, return_readout: bool = False,
                return_last_state: bool = False):
        s_tokens, ns_tokens, hist_mask = self.encode_inputs(batch, field_mask)

        s_transform = None
        if self.residual_gate is not None:
            item_emb = self.item_embedding(batch["item_id"])
            s_transform = lambda h: self.residual_gate(h, item_emb, hist_mask, gate_override)

        if return_sequence_states:
            ns_tokens, states, s_final = self.backbone(
                s_tokens, ns_tokens, hist_mask, return_states=True, s_transform=s_transform)
            if self.residual_gate is not None:
                states.append(self.residual_gate.last_zr)
        else:
            states = None
            ns_tokens, s_final = self.backbone(s_tokens, ns_tokens, hist_mask, s_transform=s_transform)

        readout = ns_tokens.flatten(start_dim=1)
        logit = self.tower(readout).squeeze(-1)
        # The reference structure discards the last layer's sequence-FFN output
        # (only NS tokens reach the head), leaving those params without any
        # autograd path. Adding `0.0 * sum` gives them a strictly-zero gradient
        # path so DDP needs no find_unused_parameters (a fragile combination
        # with gradient accumulation). This is an exact numerical no-op.
        logit = logit + 0.0 * s_final.sum()
        out = {"logit": logit, "pred": torch.sigmoid(logit)}
        if return_sequence_states:
            out["states"] = states
            out["hist_mask"] = hist_mask
        if return_readout:
            out["readout"] = readout
        if return_last_state:
            out["last_state"] = s_final
            out["hist_mask"] = hist_mask
        if return_gate and self.residual_gate is not None:
            out["gate"] = self.residual_gate.last_g
        return out

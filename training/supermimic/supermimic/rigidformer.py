# Copyright (c) MyoSuite Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0. Adapted from NDM Gym
# (github.com/Frank-ZY-Dou/ndm-gym, MIT; see LICENSE-ndm-gym.txt in training/supermimic/)
# and RigidFormer (Z. Dou et al., arXiv:2605.09196).
"""Body-token RigidFormer backbone for musculoskeletal world models.

Builds one token per skeleton body (plus register tokens) with heading-invariant 3D rotary
attention (:class:`Rope3d`), kinematic-tree hop-distance attention biases (:func:`tree_hops`),
pooled attached-muscle excitation features, and reduced-coordinate readout heads.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from supermimic.kinematics import SkeletonKinematics
from supermimic.rotations import mat_to_6d, mm3, mv3, yaw_mat

POS_SCALE = 0.1
BODY_TOKEN_DIM = 24  # pos_h (3) + rot_6d (6) + lin_vel (3) + ang_vel (3) + rel_rot_6d (6) + rel_ang_vel (3)


def tree_hops(parent: Sequence[int]) -> torch.Tensor:
    """Shortest kinematic-tree edge distance ``(B, B)`` between every pair of bodies."""
    ancestors: list[dict[int, int]] = []
    for i in range(len(parent)):
        path: dict[int, int] = {}
        j, dist = i, 0
        while j >= 0:
            path[j] = dist
            j = int(parent[j])
            dist += 1
        ancestors.append(path)
    n = len(parent)
    hops = torch.zeros((n, n), dtype=torch.long)
    for i in range(n):
        for j in range(n):
            hops[i, j] = min(
                ancestors[i][b] + d
                for b, d in ancestors[j].items()
                if b in ancestors[i]
            )
    return hops


def muscle_body_attachments(
    m: mujoco.MjModel,
    kin: SkeletonKinematics,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Row-normalized ``(B, M)`` matrix mapping muscles to the kept skeleton bodies they attach to."""
    keep_map = {m.body(name).id: idx for idx, name in enumerate(kin.body_names)}
    num_b, num_m = len(kin.body_names), int(m.nu)
    mat = np.zeros((num_b, num_m), dtype=np.float32)
    site_wrap = int(mujoco.mjtWrap.mjWRAP_SITE)
    sphere_wrap = int(mujoco.mjtWrap.mjWRAP_SPHERE)
    cyl_wrap = int(mujoco.mjtWrap.mjWRAP_CYLINDER)

    def nearest_kept(bid: int) -> int:
        cur = int(bid)
        while cur > 0 and cur not in keep_map:
            cur = int(m.body_parentid[cur])
        return keep_map.get(cur, 0)

    for col in range(num_m):
        if int(m.actuator_trntype[col]) != int(mujoco.mjtTrn.mjTRN_TENDON):
            jid = int(m.actuator_trnid[col, 0])
            if jid >= 0:
                mat[nearest_kept(int(m.jnt_bodyid[jid])), col] = 1.0
            continue
        tid = int(m.actuator_trnid[col, 0])
        adr, count = int(m.tendon_adr[tid]), int(m.tendon_num[tid])
        attached: set[int] = set()
        for w in range(adr, adr + count):
            wtype = int(m.wrap_type[w])
            obj = int(m.wrap_objid[w])
            if wtype == site_wrap:
                attached.add(nearest_kept(int(m.site_bodyid[obj])))
            elif wtype in (sphere_wrap, cyl_wrap):
                attached.add(nearest_kept(int(m.geom_bodyid[obj])))
        for b_idx in attached:
            mat[b_idx, col] = 1.0

    row_sums = np.maximum(mat.sum(axis=1, keepdims=True), 1.0)
    return torch.tensor(mat / row_sums, dtype=dtype, device=device)


class Rope3d(nn.Module):
    """3D rotary position embedding for per-token spatial coordinates."""

    def __init__(self, dim: int, max_freq: float = 10.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 6 != 0:
            raise ValueError(
                f"Rope3d requires a positive multiple of 6 channels, got {dim}"
            )
        self.dim = dim
        self.register_buffer(
            "freqs",
            torch.logspace(0.0, math.log10(max_freq), dim // 6),
            persistent=False,
        )

    def forward(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map positions ``(N, T, 3)`` to ``(cos, sin)`` each of shape ``(N, 1, T, dim // 2)``."""
        ang = (pos.float().unsqueeze(-1) * self.freqs).flatten(-2)
        return ang.cos().unsqueeze(1), ang.sin().unsqueeze(1)


def rope_rotate_fused(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Elementwise 3D rotary rotation of leading ``2 * cos.shape[-1]`` head channels."""
    pad = x.shape[-1] - 2 * cos.shape[-1]
    c = F.pad(cos.repeat_interleave(2, -1), (0, pad), value=1.0)
    s = F.pad(torch.stack((-sin, sin), -1).flatten(-2), (0, pad))
    xf = x.float()
    out = xf * c + xf.unflatten(-1, (-1, 2)).flip(-1).flatten(-2) * s
    return out.to(x.dtype)


class RigidFormerBlock(nn.Module):
    """Pre-norm gated-attention + SwiGLU Transformer block with QK-RMSNorm and 3D RoPE."""

    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_mult: float = 2.0,
        ctx_dim: int = 0,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.hd = dim // heads
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.qkvg = nn.Linear(dim, 4 * dim)
        self.q_norm = nn.RMSNorm(self.hd, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.hd, eps=1e-6)
        self.proj = nn.Linear(dim, dim)
        self.film: nn.Linear | None = None
        if ctx_dim > 0:
            self.film = nn.Linear(ctx_dim, 2 * dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        hidden = int(ffn_mult * dim)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_batch, n_tok, dim = x.shape
        q, k, v, g = (
            self.qkvg(self.norm1(x))
            .view(n_batch, n_tok, 4, self.heads, self.hd)
            .unbind(2)
        )
        dt = self.q_norm.weight.dtype
        q = self.q_norm(q.to(dt)).transpose(1, 2)
        k = self.k_norm(k.to(dt)).transpose(1, 2)
        v = v.transpose(1, 2)
        if cos is not None and sin is not None:
            q = rope_rotate_fused(q, cos, sin).to(v.dtype)
            k = rope_rotate_fused(k, cos, sin).to(v.dtype)
        else:
            q, k = q.to(v.dtype), k.to(v.dtype)
        attn_mask = None if bias is None else bias.to(v.dtype)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.proj(
            a.transpose(1, 2).reshape(n_batch, n_tok, dim)
            * torch.sigmoid(g.reshape(n_batch, n_tok, dim))
        )
        if self.film is not None and ctx is not None:
            gamma, beta = self.film(ctx).unsqueeze(1).chunk(2, dim=-1)
            x = x * (1.0 + gamma) + beta
        u, gate = self.w12(self.norm2(x)).chunk(2, dim=-1)
        return x + self.w3(F.silu(gate) * u)


class RigidFormerBackbone(nn.Module):
    """Articulated body-token Transformer backbone for :class:`~supermimic.networks.WorldModel`."""

    def __init__(
        self,
        nq: int,
        num_muscles: int,
        root_out: int,
        joint_out: int,
        num_touch: int,
        *,
        kin: SkeletonKinematics | None = None,
        mj_model: mujoco.MjModel | None = None,
        dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        registers: int = 2,
        muscle_dim: int = 32,
        ffn_mult: float = 2.0,
        max_freq: float = 10.0,
    ) -> None:
        super().__init__()
        if registers < 1:
            raise ValueError("registers must be >= 1")
        self.nq = nq
        self.num_muscles = num_muscles
        self.root_out = root_out
        self.joint_out = joint_out
        self.num_touch = num_touch
        self.dim = dim
        self.heads = heads
        self.registers = registers
        self.kin = kin

        hd = dim // heads
        rope_dim = (hd // 6) * 6
        self.rope = Rope3d(rope_dim, max_freq=max_freq) if rope_dim >= 6 else None

        if kin is not None:
            self.num_bodies = kin.num_bodies
            self.body_feat_dim = BODY_TOKEN_DIM
            parent = list(kin.parent)
            parent_clamped = [max(p, 0) for p in parent]
            self.register_buffer(
                "parent_idx",
                torch.tensor(parent_clamped, dtype=torch.long),
                persistent=False,
            )
            hops = tree_hops(parent)
            max_hop = int(hops.max().item()) if hops.numel() > 0 else 0
            tot_tok = self.num_bodies + registers
            rel = torch.full((tot_tok, tot_tok), max_hop + 1, dtype=torch.long)
            rel[: self.num_bodies, : self.num_bodies] = hops
            self.register_buffer("rel_idx", rel, persistent=False)
            self.n_rel = max_hop + 2

            # Map each independent joint column to the body index that carries it.
            joint_body = [0] * nq
            for b_idx, jlist in enumerate(kin.joints):
                for jt in jlist:
                    src_kind, col = jt["src"]
                    if src_kind == "q" and 0 <= col < nq:
                        joint_body[col] = b_idx
            self.register_buffer(
                "joint_body_idx",
                torch.tensor(joint_body, dtype=torch.long),
                persistent=False,
            )
            if mj_model is not None:
                attach = muscle_body_attachments(mj_model, kin, device=kin.off_p.device)
            else:
                attach = torch.full(
                    (self.num_bodies, max(num_muscles, 1)), 1.0 / max(num_muscles, 1)
                )
            self.register_buffer("muscle_attach", attach, persistent=False)
        else:
            # Standalone kinematics-free fallback: 1 root token + nq joint tokens.
            self.num_bodies = 1 + nq
            self.body_feat_dim = 13
            tot_tok = self.num_bodies + registers
            rel = torch.ones((tot_tok, tot_tok), dtype=torch.long)
            rel.fill_diagonal_(0)
            rel[self.num_bodies :, :] = 2
            rel[:, self.num_bodies :] = 2
            self.register_buffer("rel_idx", rel, persistent=False)
            self.n_rel = 3
            self.register_buffer(
                "joint_body_idx",
                torch.arange(1, 1 + nq, dtype=torch.long),
                persistent=False,
            )
            attach = torch.full(
                (self.num_bodies, max(num_muscles, 1)), 1.0 / max(num_muscles, 1)
            )
            self.register_buffer("muscle_attach", attach, persistent=False)

        self.feat_dim = self.num_bodies * self.body_feat_dim + 2 * num_muscles
        self.in_dim = self.feat_dim + 3 * self.num_bodies

        self.body_proj = nn.Linear(self.body_feat_dim, dim)
        self.body_emb = nn.Parameter(torch.zeros(self.num_bodies, dim))
        nn.init.normal_(self.body_emb, std=0.02)

        self.muscle_pair_proj = nn.Linear(2, muscle_dim)
        self.muscle_emb = nn.Parameter(torch.zeros(max(num_muscles, 1), muscle_dim))
        nn.init.normal_(self.muscle_emb, std=0.02)
        self.muscle_to_body = nn.Linear(muscle_dim, dim)
        self.reg_tokens = nn.Parameter(torch.zeros(registers, dim))
        nn.init.normal_(self.reg_tokens, std=0.02)
        self.muscle_to_ctx = nn.Linear(muscle_dim, dim)

        self.hop_bias = nn.Embedding(self.n_rel, heads)
        nn.init.zeros_(self.hop_bias.weight)

        self.blocks = nn.ModuleList(
            [
                RigidFormerBlock(dim, heads, ffn_mult=ffn_mult, ctx_dim=dim)
                for _ in range(layers)
            ]
        )
        self.out_norm = nn.RMSNorm(dim, eps=1e-6)
        self.root_head = nn.Linear(dim, root_out)
        self.joint_head = nn.Linear(dim, joint_out)
        self.joint_bias = nn.Parameter(torch.zeros(nq, joint_out))
        self.touch_head = nn.Linear(dim, num_touch) if num_touch > 0 else None
        for head in (self.root_head, self.joint_head, self.touch_head):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def hidden_matrices(self) -> list[nn.Parameter]:
        """2D Transformer block matrices optimized with Muon when ``wm_opt='muon'``."""
        mats: list[nn.Parameter] = []
        for blk in self.blocks:
            mats.extend(
                [blk.qkvg.weight, blk.proj.weight, blk.w12.weight, blk.w3.weight]
            )
        return mats

    def raw_inputs(
        self, S: tuple, act: torch.Tensor, act1: torch.Tensor
    ) -> torch.Tensor:
        """Build heading-invariant features ``(..., feat_dim)`` + scaled 3D RoPE coords ``(..., 3 * B)``."""
        p, R, v, w, q, qd = S[:6]
        lead = p.shape[:-1]
        flat_n = int(math.prod(lead))
        p_f = p.reshape(flat_n, 3)
        R_f = R.reshape(flat_n, 3, 3)
        v_f = v.reshape(flat_n, 3)
        w_f = w.reshape(flat_n, 3)
        q_f = q.reshape(flat_n, self.nq)
        qd_f = qd.reshape(flat_n, self.nq)
        H = yaw_mat(R_f)
        Ht = H.transpose(-1, -2)

        if self.kin is not None:
            if self.kin.off_p.device != p_f.device:
                # Keep kinematics buffers aligned if the model moved devices.
                self.kin.off_p = self.kin.off_p.to(p_f.device)
                self.kin.off_R = self.kin.off_R.to(p_f.device)
                self.kin.R_off = self.kin.R_off.to(p_f.device)
            P, Rb, V, W = self.kin.forward(
                p_f, R_f, v_f, w_f, q_f, qd_f, all_bodies=True
            )
            origin_xy = torch.cat(
                (p_f[:, :2], torch.zeros_like(p_f[:, 2:3])), dim=-1
            ).unsqueeze(1)
            pos_h = mv3(Ht.unsqueeze(1), P - origin_xy)
            rot_h = mat_to_6d(mm3(Ht.unsqueeze(1), Rb))
            vel_h = mv3(Ht.unsqueeze(1), V)
            ang_h = mv3(Ht.unsqueeze(1), W)
            R_par = Rb[:, self.parent_idx]
            W_par = W[:, self.parent_idx]
            rel_rot = mat_to_6d(mm3(R_par.transpose(-1, -2), Rb))
            rel_ang = mv3(Rb.transpose(-1, -2), W - W_par)
            # Root body has no parent: zero its relative parent rotation offset and rate.
            rel_rot = rel_rot.clone()
            rel_ang = rel_ang.clone()
            rel_rot[:, 0] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=p_f.device, dtype=p_f.dtype
            )
            rel_ang[:, 0] = 0.0
            body_feats = torch.cat(
                (pos_h, rot_h, vel_h, ang_h, rel_rot, rel_ang), dim=-1
            )
            rope_pos = pos_h / POS_SCALE
        else:
            Rt = R_f.transpose(-1, -2)
            root_feat = torch.cat(
                (p_f[:, 2:3], mat_to_6d(mm3(Ht, R_f)), mv3(Rt, v_f), mv3(Rt, w_f)),
                dim=-1,
            )
            j_feats = torch.zeros(
                (flat_n, self.nq, 13), device=p_f.device, dtype=p_f.dtype
            )
            j_feats[:, :, 0] = q_f
            j_feats[:, :, 1] = qd_f
            body_feats = torch.cat((root_feat.unsqueeze(1), j_feats), dim=1)
            rope_pos = torch.zeros(
                (flat_n, self.num_bodies, 3), device=p_f.device, dtype=p_f.dtype
            )
            rope_pos[:, :, 2] = p_f[:, 2:3] / POS_SCALE

        feats = torch.cat(
            (
                body_feats.reshape(flat_n, -1),
                act.reshape(flat_n, self.num_muscles),
                act1.reshape(flat_n, self.num_muscles),
                rope_pos.reshape(flat_n, -1),
            ),
            dim=-1,
        )
        return feats.reshape(*lead, self.in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        flat_n = int(math.prod(lead))
        xf = x.reshape(flat_n, self.in_dim)

        b_end = self.num_bodies * self.body_feat_dim
        m_end = b_end + 2 * self.num_muscles
        b_in = xf[:, :b_end].view(flat_n, self.num_bodies, self.body_feat_dim)
        m_in = xf[:, b_end:m_end].view(flat_n, self.num_muscles, 2)
        rope_body = xf[:, m_end:].view(flat_n, self.num_bodies, 3)

        tok = self.body_proj(b_in) + self.body_emb.unsqueeze(0)
        if self.num_muscles > 0:
            m_enc = F.silu(
                self.muscle_pair_proj(m_in)
                + self.muscle_emb[: self.num_muscles].unsqueeze(0)
            )
            attach = self.muscle_attach[:, : self.num_muscles].to(m_enc.dtype)
            pooled = torch.einsum("bm,nmd->nbd", attach, m_enc)
            tok = tok + self.muscle_to_body(pooled)
            ctx = self.muscle_to_ctx(m_enc.mean(dim=1))
        else:
            ctx = tok.new_zeros((flat_n, self.dim))

        reg = self.reg_tokens.unsqueeze(0).expand(flat_n, -1, -1)
        tokens = torch.cat((tok, reg), dim=1)

        cos = sin = None
        if self.rope is not None:
            reg_pos = rope_body[:, :1, :].expand(flat_n, self.registers, 3).clone()
            reg_pos[:, :, :2] = 0.0
            cos, sin = self.rope(torch.cat((rope_body, reg_pos), dim=1))

        bias = self.hop_bias(self.rel_idx).permute(2, 0, 1).unsqueeze(0)
        for blk in self.blocks:
            tokens = blk(tokens, cos=cos, sin=sin, bias=bias, ctx=ctx)
        tokens = self.out_norm(tokens)

        root_y = self.root_head(tokens[:, 0])
        j_tok = tokens[:, self.joint_body_idx]
        j_y = self.joint_head(j_tok) + self.joint_bias.unsqueeze(0)
        if self.joint_out == 1:
            joint_flat = j_y.squeeze(-1)
        else:
            # Match WorldModel output order: all dq (nq) then all dqd (nq).
            joint_flat = j_y.transpose(1, 2).reshape(flat_n, self.nq * self.joint_out)

        parts = [root_y, joint_flat]
        if self.touch_head is not None:
            parts.append(self.touch_head(tokens[:, self.num_bodies]))
        out = torch.cat(parts, dim=-1)
        return out.reshape(*lead, -1)

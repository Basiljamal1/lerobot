from __future__ import annotations

import logging
import threading
import time
from queue import Queue
from typing import Any

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class RTCPolicyFlowModel:
    """
    Flow-model wrapper that runs Algorithm 1 in a background thread.

    Composition-only:
      - wraps a *flow model* instance (e.g., VLAFlowMatching)
      - does not know about the inference policy class
      - owns input/output queues and guided inpainting
      - proxies unknown attributes to the underlying flow model
    """

    def __init__(
        self,
        *,
        flow_model: nn.Module,  # e.g., VLAFlowMatching
        chunk_size: int,  # H
        num_steps: int,  # n
        beta: float,  # β (guidance clip)
        device: torch.device,
        use_cache: bool = True,
    ):
        self.flow_model = flow_model
        self.H = int(chunk_size)
        self.n = int(num_steps)
        self.beta = float(beta)
        self.device = device
        self.use_cache = bool(use_cache)

        # Queues for the background thread
        # prepared_inputs := dict(images, img_masks, lang_tokens, lang_masks, state)
        self.input_queue: Queue[tuple[dict[str, Tensor], int, int]] = Queue()  # (prepared_inputs, s, d)
        self.output_queue: Queue[Tensor] = Queue()  # A_new [B,H,D]

        self.current_action: Tensor | None = None
        self._thread: threading.Thread | None = None

        # Freeze backbone (inference only)
        # TODO: Refactor to make this model agnostic.
        vlm = getattr(self.flow_model, "vlm_with_expert", None)
        if vlm is not None:
            for p in vlm.parameters():
                p.requires_grad = False

    # ---- Attribute proxy (so downstream code still finds tokenizer etc.) ----
    def __getattr__(self, name: str):
        return getattr(self.flow_model, name)

    # For callers that used .get_vlm() in older code
    # TODO: Refactor to make this model agnostic.
    # smolvla uses vlm_with_expert; pi0 uses paligemma_with_expert
    def get_vlm(self):
        return self.flow_model.vlm_with_expert

    # ---- Lifecycle ----
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def start_inference_loop(self):
        self.start()

    def reset(self):
        self.current_action = None

    # ---- Background loop (Algorithm 1 outer) ----
    def _loop(self):
        while True:
            prepared_inputs, s, d = self.input_queue.get()

            images = list(prepared_inputs["images"])
            img_masks = list(prepared_inputs["img_masks"])
            lang_tokens = prepared_inputs["lang_tokens"].clone()
            lang_masks = prepared_inputs["lang_masks"].clone()
            state = prepared_inputs["state"].clone()

            t0 = time.time()
            if self.current_action is None:
                # First-time: unguided sample A ~ π(·|o)
                self.current_action = self.flow_model.sample_actions(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                )  # [B,H,D]

                logger.debug(f"[RTC] unguided sample time: {time.time() - t0:.3f}s")
            else:
                previous_action = self.current_action[:, s:, :]  # A_prev = A_cur[s:H]
                self.current_action = self._guided_inference(
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    previous_action=previous_action,
                    delay=d,
                    start_horizon=s,
                )
                logger.debug(f"[RTC] Guided inpainting time: {time.time() - t0:.3f}s")
            self.output_queue.put(self.current_action)

    # ---- ΠGDM guided inpainting (Eq. 1–5) ----
    def _guided_inference(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        previous_action: Tensor,
        delay: int,
        start_horizon: int,
    ) -> Tensor:
        # Prefix & KV cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.flow_model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        # Model-agnostic attempt to get the attn mask builder
        make_att_2d_masks = getattr(self.flow_model, "make_att_2d_masks", None)
        if make_att_2d_masks is None:
            # Fallback to smolVLA helper (safe for smolVLA; adapt per other models as needed)
            from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks  # type: ignore
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, kv = self.flow_model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.use_cache,
            fill_kv_cache=True,
        )

        # Right-pad A_prev to length H
        batch_dim, action_dim = previous_action.shape[0], previous_action.shape[-1]
        if previous_action.shape[1] < self.H:
            pad = torch.zeros(
                batch_dim,
                self.H - previous_action.shape[1],
                action_dim,
                device=self.device,
                dtype=previous_action.dtype,
            )
            previous_action = torch.cat([previous_action, pad], dim=1)

        # Soft mask W (Eq. 5) – shape [1,H,1]
        weights = self._soft_mask_exact(horizon=self.H, s=start_horizon, d=delay, device=self.device)

        # Integrate Eq. (1) for n steps
        a_t = self.flow_model.sample_noise((batch_dim, self.H, action_dim), self.device)
        dt, eps = 1.0 / float(self.n), 1e-8

        for i in range(self.n):
            tau = float(i) / float(self.n)
            tau_t = torch.full((batch_dim,), tau, device=self.device, dtype=a_t.dtype)

            # vπ(A, o, τ) and f(A)=A+(1-τ)vπ (Eq. 3)
            v_t = self._velocity(a_t, tau_t, kv, prefix_pad_masks)
            f_at = a_t + (1.0 - tau) * v_t

            # masked residual
            e = (previous_action - f_at) * weights

            # VJP g = J^T e for f at A_t (Eq. 2)
            def f_denoise(a_t: Tensor, tau=tau, tau_t=tau_t) -> Tensor:
                v = self._velocity(a_t, tau_t, kv, prefix_pad_masks)
                return a_t + (1.0 - tau) * v

            _, g = torch.autograd.functional.vjp(f_denoise, a_t.requires_grad_(True), v=e, create_graph=False)

            # guidance weight clip (Eq. 2 & 4)
            r2 = ((1.0 - tau) ** 2) / (tau**2 + (1.0 - tau) ** 2 + eps)
            w = min(self.beta, (1.0 - tau) / (max(tau, eps) * r2 + eps))
            w = torch.tensor(w, device=self.device, dtype=a_t.dtype).view(1, 1, 1)

            # Euler step (Eq. 1)
            a_t = a_t + dt * (v_t + w * g)

        return a_t

    @staticmethod
    def _soft_mask_exact(horizon: int, s: int, d: int, device: torch.device) -> Tensor:
        weights = torch.zeros(horizon, device=device, dtype=torch.float32)
        if d > 0:
            weights[:d] = 1.0
        overlap_end = max(0, horizon - s)
        if d < overlap_end:
            mask = torch.arange(d, overlap_end, device=device, dtype=torch.float32)
            denom = horizon - s - d + 1
            c = (horizon - s - mask) / denom
            weights[d:overlap_end] = (
                c * torch.exp(c - 1.0) / (torch.exp(torch.tensor(1.0, device=device)) - 1.0)
            )
        return weights.view(1, horizon, 1)

    @torch.no_grad()
    def _velocity(self, a_t: Tensor, tau_t: Tensor, kv: Any, prefix_pad_masks: Tensor) -> Tensor:
        # Suffix path with cached prefix
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.flow_model.embed_suffix(a_t, tau_t)
        make_att_2d_masks = getattr(self.flow_model, "make_att_2d_masks", None)
        if make_att_2d_masks is None:
            from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks  # type: ignore
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        batch_dim_size = prefix_pad_masks.shape[0]
        l_p = prefix_pad_masks.shape[1]
        l_s = suffix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(batch_dim_size, l_s, l_p)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        pos_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        with torch.no_grad():
            outputs_embeds, _ = self.flow_model.vlm_with_expert.forward(
                attention_mask=full_att_2d,
                position_ids=pos_ids,
                past_key_values=kv,
                inputs_embeds=[None, suffix_embs],
                use_cache=self.use_cache,
                fill_kv_cache=False,
            )
            suffix_out = outputs_embeds[1][:, -self.H :].to(dtype=torch.float32)

        return self.flow_model.action_out_proj(suffix_out)

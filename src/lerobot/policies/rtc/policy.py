from __future__ import annotations

from collections import deque
from queue import Empty

import torch
from torch import Tensor

from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .model_wrapper import RTCPolicyFlowModel


class RTCPolicy:
    """
    Policy wrapper that manages:
      • scheduling (s, d, delay buffer Q)
      • action queue and swap-as-soon-as-ready
      • observation prep and post-processing via the *inference model*
      • background chunking via RTCPolicyFlowModel (flow wrapper)

    Composition:
      self.inference_model  -> e.g., SmolVLAPolicy  (no queues)
      self.model            -> RTCPolicyFlowModel   (has queues/thread; proxies flow model attrs)
    """

    name = "rtc_policy"

    def __init__(self, *, config, inference_model, flow_model: RTCPolicyFlowModel):
        self.config = config
        self.inference_model = inference_model
        self.model = flow_model  # expose as .model so downstream code finds tokenizer via proxy
        self.model.start()

        # Horizons
        self.H = int(self.config.chunk_size)
        self.exec_len = int(self.config.n_action_steps)

        # Scheduling params (Algorithm 1)
        self.s_min = int(
            getattr(
                self.config, "s_min", max(1, min(self.exec_len, getattr(self.config, "inference_steps", 1)))
            )
        )
        self.delay_buffer_size = int(getattr(self.config, "delay_buffer_size", 10))
        self.initial_delay = int(
            getattr(self.config, "initial_delay", getattr(self.config, "inference_steps", self.s_min))
        )
        self.delay_buffer: deque[int] = deque([self.initial_delay], maxlen=self.delay_buffer_size)

        # RTC state
        self.since_start = 0
        self.inference_in_flight = False
        self.s_target: int | None = None
        self.needs_init_chunk = True

        # Public action queue
        self._queues: dict[str, deque] = {ACTION: deque()}

    # ---- Policy surface ----
    def eval(self):
        if hasattr(self.inference_model, "eval"):
            self.inference_model.eval()
        return self

    def reset(self):
        self.since_start = 0
        self.inference_in_flight = False
        self.s_target = None
        self.needs_init_chunk = True
        self.delay_buffer.clear()
        self.delay_buffer.append(int(self.initial_delay))
        self._queues[ACTION].clear()
        self.model.reset()
        if hasattr(self.inference_model, "reset"):
            self.inference_model.reset()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        # Let the inference policy do its usual prep (normalization, device, etc.)
        prepared = batch
        if hasattr(self.inference_model, "_prepare_batch"):
            prepared = self.inference_model._prepare_batch(batch)

        # Keep observation history queues (except ACTION)
        self._queues = populate_queues(self.inference_model._queues, prepared, exclude_keys=[ACTION])

        # Swap to the next chunk if it's ready
        self._maybe_swap_to_ready_chunk()

        # Kick off the next inference exactly at t == s (or at init)
        self._maybe_start_next_inference(prepared)

        # Cold start: block once to fill the queue
        if len(self._queues[ACTION]) == 0:
            self._blocking_fill_from_next_chunk(prepared)

        # Pop one action for execution
        action = self._queues[ACTION].popleft()
        self.since_start += 1
        return action

    # ---- Helpers: Algorithm 1 semantics ----
    @torch.no_grad()
    def _maybe_start_next_inference(self, batch_prepared: dict[str, Tensor]):
        if self.inference_in_flight:
            return

        d_est = max(self.delay_buffer) if len(self.delay_buffer) else self.initial_delay
        # s = max(d, s_min), but keep at least one un-constrained step at the tail
        s = int(min(self.exec_len - 1, max(d_est, self.s_min)))

        if self.since_start == s or self.needs_init_chunk:
            prepared_inputs = self._build_prepared_inputs(batch_prepared)
            self.model.input_queue.put((prepared_inputs, s, d_est))
            self.s_target = s
            self.inference_in_flight = True
            self.needs_init_chunk = False

    @torch.no_grad()
    def _maybe_swap_to_ready_chunk(self):
        if not self.inference_in_flight:
            return
        try:
            a_new = self.model.output_queue.get_nowait()
        except Empty:
            return

        # Observed delay δ = t - s
        delta = max(0, self.since_start - int(self.s_target or 0))
        self.delay_buffer.append(int(delta))

        # Post-process to env action space and re-index so next action is A_new[δ]
        actions = self._postprocess_chunk(a_new)  # [B,H,D*]
        tail = actions.transpose(0, 1)[delta : delta + self.exec_len]
        self._queues[ACTION].clear()
        self._queues[ACTION].extend(tail)

        # Reset t so that it now indexes into A_new
        self.since_start = int(delta)
        self.inference_in_flight = False
        self.s_target = None

    @torch.no_grad()
    def _blocking_fill_from_next_chunk(self, batch_prepared: dict[str, Tensor]):
        if not self.inference_in_flight:
            prepared_inputs = self._build_prepared_inputs(batch_prepared)
            self.model.input_queue.put((prepared_inputs, 0, max(self.delay_buffer)))
            self.s_target = 0
            self.inference_in_flight = True
            self.needs_init_chunk = False

        a_new = self.model.output_queue.get()
        self.delay_buffer.append(0)
        actions = self._postprocess_chunk(a_new)
        self._queues[ACTION].extend(actions.transpose(0, 1)[: self.exec_len])

        self.since_start = 0
        self.inference_in_flight = False
        self.s_target = None

    # ---- Obs build & post-processing via the *inference model* ----
    @torch.no_grad()
    def _build_prepared_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Stack any queued history tensors kept in RTCPolicy (not in inference model)
        for k in list(batch.keys()):
            if k in self._queues:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        # Use the inference policy's own preprocessors (model-agnostic)
        images, img_masks = self.inference_model.prepare_images(batch)
        state = self.inference_model.prepare_state(batch)

        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        return {
            "images": images,
            "img_masks": img_masks,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "state": state,
        }

    @torch.no_grad()
    def _postprocess_chunk(self, actions: Tensor) -> Tensor:
        # Trim to original action dim if present on config
        if hasattr(self.config, "action_feature") and self.config.action_feature is not None:
            dim = int(self.config.action_feature.shape[0])
            actions = actions[:, :, :dim]

        # Unnormalize via inference model if available
        if hasattr(self.inference_model, "unnormalize_outputs"):
            actions = self.inference_model.unnormalize_outputs({ACTION: actions})[ACTION]

        # Optional π-ALOHA encoding
        if getattr(self.config, "adapt_to_pi_aloha", False) and hasattr(
            self.inference_model, "_pi_aloha_encode_actions"
        ):
            actions = self.inference_model._pi_aloha_encode_actions(actions)

        return actions

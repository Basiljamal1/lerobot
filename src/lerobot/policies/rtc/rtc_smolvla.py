# lerobot/policies/smolvla/policy_rtc_paper.py
from __future__ import annotations

import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc_smolvla import RTCSmolVLACofnig
from lerobot.policies.rtc.model_wrapper import RTCPolicyFlowModel
from lerobot.policies.rtc.policy import RTCPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # the stock policy (inference model)


class RTCSmolVLAPolicy(PreTrainedPolicy):
    """
    Thin shim that *contains* RTCPolicy and relays calls.
    name = "rtc_smolvla", config_class = RTCSmolVLACofnig
    """

    name = "rtc_smolvla"
    config_class = RTCSmolVLACofnig

    def __init__(self, config: RTCSmolVLACofnig):
        super().__init__(config)
        config.validate_features()

        # 1) Build the stock SmolVLA inference policy (no queues here)
        self.inference_model = SmolVLAPolicy(config)  # has .model = VLAFlowMatching

        # 2) Wrap its *flow model* with RTCPolicyFlowModel (queues & thread live here)
        flow = RTCPolicyFlowModel(
            flow_model=self.inference_model.model,
            chunk_size=int(config.chunk_size),
            num_steps=int(config.num_steps),
            beta=float(config.beta),
            device=torch.device(config.device),
            use_cache=getattr(config, "use_cache", True),
        )

        # 3) Build the RTCPolicy (scheduling, action queue, swap)
        self._rtc = RTCPolicy(config=config, inference_model=self.inference_model, flow_model=flow)

        # 4) Important: expose the flow wrapper as .model so user code can find tokenizer etc.
        self.model = self._rtc.model

    # ---- Relay API ----
    def reset(self):
        self._rtc.reset()

    def eval(self):
        self._rtc.eval()
        return self

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        # Reuse SmolVLA preprocessing, then relay to RTCPolicy
        if hasattr(self.inference_model, "_prepare_batch"):
            batch = self.inference_model._prepare_batch(batch)
        return self._rtc.select_action(batch)

    def get_optim_params(self):
        return self.inference_model.get_optim_params()

    def forward(self, batch):
        return self.inference_model.forward(batch)

    def predict_action_chunk(self, batch):
        return self.inference_model.predict_action_chunk(batch)

"""Importable Verl class for the simplified PPP agent-loop registration.

This module is deliberately separate from :mod:`ppp_simplified.verl_loop`.
Hydra resolves agent-loop classes by fully qualified name inside Ray workers,
so the class must live at module scope. Laptop-side helpers can still import
``verl_loop`` without importing Torch, Ray, or Verl.
"""

from __future__ import annotations

from typing import Any

from agents.utils import CallLLM, TaskContext
from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopBase

from .verl_loop import run_simplified_rollout


class SimplifiedPPPAgentLoop(AgentLoopBase):
    """Ray-worker entry point for one navigation-v2 PPP trajectory."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls.config = config
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls._class_initialized = True

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        item = DataProto.from_dict(non_tensors=kwargs)
        llm_client = CallLLM(
            url=self.server_manager,
            tokenizer=self.tokenizer,
            config=self.config.actor_rollout_ref.rollout,
            loop=self.loop,
        )
        context = TaskContext(
            config=self.config,
            global_step=kwargs.get(
                "global_steps",
                kwargs.get("global_step", 0),
            ),
            is_train=kwargs.get("is_train", True),
            tokenizer=self.tokenizer,
            llm_client=llm_client,
        )
        return await run_simplified_rollout(
            item=item,
            context=context,
            server_manager=self.server_manager,
        )

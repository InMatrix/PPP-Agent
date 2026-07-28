"""Teaching-scale PPP RL entry point using the simplified read-only agent.

Ray workers load the importable loop class through
``simplified/config/verl_agent_loops.yaml``. Keeping registration in the
Hydra configuration ensures every worker sees it, not only this driver.
"""


def main() -> None:
    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()

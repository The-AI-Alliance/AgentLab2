import logging
import os
import time

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.envs.browser import BrowserEnvConfig
from agentlab2.experiment import Experiment
from agentlab2.llm import LLM

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d %(funcName)s() - %(message)s",
)

debug = False


def main():
    current_datetime = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.expanduser(f"~/agentlab_results/al2/miniwob_{current_datetime}")

    llm = LLM(model_name="azure/gpt-5-mini", temperature=1.0)
    env_config = BrowserEnvConfig(headless=True, use_screenshot=True)
    agent_config = ReactAgentConfig(llm=llm)
    benchmark = MiniWobBenchmark(env_config=env_config)
    exp = Experiment(name="miniwob", output_dir=output_dir, agent_config=agent_config, benchmark=benchmark)

    if debug:
        exp.run_sequential(debug_limit=2)
    else:
        exp.run_ray(n_cpus=4)


if __name__ == "__main__":
    main()

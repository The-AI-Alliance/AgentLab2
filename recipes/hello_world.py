import logging
import os

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.envs.browser import BrowserEnvConfig
from agentlab2.experiment import Experiment
from agentlab2.llm import LLM

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d %(funcName)s() - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


def main():
    miniwob_dir = os.path.expanduser("~/miniwob-plusplus")
    llm = LLM(model_name="azure/gpt-5-mini", temperature=1.0)
    env_config = BrowserEnvConfig(headless=True, timeout=30000)
    agent_config = ReactAgentConfig(llm=llm, use_html=True, use_screenshot=True)
    benchmark = MiniWobBenchmark(dataset_dir=miniwob_dir)
    exp = Experiment(
        name="hello_world_study",
        output_dir="./hello_world_1",
        env_config=env_config,
        agent_config=agent_config,
        benchmark=benchmark,
    )
    exp.run_sequential(save_results=True, debug_limit=1)


if __name__ == "__main__":
    main()

import os

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.envs.browser import BrowserEnvConfig
from agentlab2.experiment import Experiment
from agentlab2.llm import LLM


def main():
    llm = LLM(model_name="azure/gpt-5-mini", temperature=1.0)
    env_config = BrowserEnvConfig(headless=True, timeout=30000)
    agent_config = ReactAgentConfig(use_html=True, use_screenshot=True)
    benchmark = MiniWobBenchmark(dataset_dir=os.path.expanduser("~/miniwob-plusplus"))
    benchmark.prepare()  # Start the MiniWob server
    exp = Experiment(
        name="hello_world_study",
        output_dir="./hello_world_1",
        llm=llm,
        env_config=env_config,
        agent_config=agent_config,
        benchmark=benchmark,
    )
    # exp.run_sequential(save_results=True)
    runs = exp.get_runs()
    runs[0].run()


if __name__ == "__main__":
    main()

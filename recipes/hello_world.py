from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.envs.browser import BrowserEnvConfig
from agentlab2.experiment import Experiment
from agentlab2.llm import LLM


def main():
    llm = LLM(model_name="azure/gpt-5-mini", temperature=1.0, max_total_tokens=128000)
    env_config = BrowserEnvConfig(headless=True, timeout=30000)
    agent_config = ReactAgentConfig(use_html=True, use_screenshot=True)
    benchmark = MiniWobBenchmark(remove_human_display=True, episode_max_time=1000000)
    study = Experiment(
        name="hello_world_study",
        llm=llm,
        env_config=env_config,
        agent_config=agent_config,
        benchmark=benchmark,
    )
    study.run_sequential()


if __name__ == "__main__":
    main()

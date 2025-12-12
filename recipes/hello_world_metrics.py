import logging
import os

from agentlab2.agents.react import ReactAgentConfig
from agentlab2.benchmarks.miniwob.benchmark import MiniWobBenchmark
from agentlab2.envs.browser import BrowserEnvConfig
from agentlab2.experiment import Experiment
from agentlab2.llm import LLM
from agentlab2.metrics.tracer import AgentTracer


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s:%(lineno)d %(funcName)s() - %(message)s",
)


def main() -> None:
    miniwob_dir = os.path.expanduser("~/miniwob-plusplus")
    llm = LLM(model_name="gpt-5-mini", temperature=1.0)
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

    tracer = AgentTracer(service_name=exp.name, output_dir=exp.output_dir)
    logging.info(f"Metrics run_id={tracer.run_id}, output={tracer.output_dir}")

    try:
        with tracer.benchmark(exp.name):
            exp.benchmark.setup()
            runs = exp.create_runs()[:1]

            traces = []
            for run in runs:
                with tracer.episode(run.task.id):
                    trace = run.run()
                    traces.append(trace)

            exp.save_traces(traces)
    finally:
        exp.benchmark.close()
        tracer.shutdown()


if __name__ == "__main__":
    main()

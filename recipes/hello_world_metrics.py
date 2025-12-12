import logging
import os
import uuid

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
    miniwob_dir = os.path.expanduser("~/projects/miniwob-plusplus")
    llm = LLM(model_name="gpt-5-mini", temperature=1.0)
    env_config = BrowserEnvConfig(headless=True, timeout=30000, pw_kwargs={"chromium_sandbox": False})
    agent_config = ReactAgentConfig(llm=llm, use_html=True, use_screenshot=True)
    benchmark = MiniWobBenchmark(dataset_dir=miniwob_dir, env_config=env_config)
    exp = Experiment(
        name="hello_world_study",
        output_dir="./hello_world_1",
        agent_config=agent_config,
        benchmark=benchmark,
    )

    trace_output = f"{exp.output_dir}/metrics/{uuid.uuid4()}"
    tracer = AgentTracer(
        service_name=exp.name,
        output_dir=trace_output,
        otlp_endpoint="http://localhost:4318/v1/traces",
    )
    logging.info(f"Metrics output={tracer.output_dir}")

    try:
        with tracer.benchmark(exp.name):
            trajectories = exp.run_ray(n_cpus=4)

        logging.info(f"Completed {len(trajectories)} trajectories")
    finally:
        tracer.shutdown()


if __name__ == "__main__":
    main()

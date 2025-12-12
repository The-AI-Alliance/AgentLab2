"""Standalone demo of the OTEL-native metrics tracing system.

Demonstrates AgentTracer usage without requiring the full agent framework.
Shows benchmark/episode/step hierarchy and automatic episode export.
"""

import uuid

from agentlab2.metrics.tracer import AgentTracer


def main() -> None:
    tracer = AgentTracer(service_name="demo-agent", output_dir=f"./demo_output/metrics/{uuid.uuid4()}")
    print(f"Output: {tracer.output_dir}")

    with tracer.benchmark("demo_experiment"):
        for ep_num in range(3):
            with tracer.episode(f"episode_{ep_num}"):
                for step_num in range(5):
                    tracer.log({"action": f"action_{step_num}", "reward": step_num * 0.1})

    tracer.shutdown()

    print("\nOutput structure:")
    for path in sorted(tracer.output_dir.rglob("*")):
        rel = path.relative_to(tracer.output_dir)
        indent = "  " * len(rel.parts)
        print(f"{indent}{path.name}")


if __name__ == "__main__":
    main()

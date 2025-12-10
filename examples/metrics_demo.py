import os
import tempfile

from agentlab2.metrics.tracer import AgentTracer


def main() -> None:
    with tempfile.TemporaryDirectory() as run_dir:
        print(f"Run dir: {run_dir}")
        tracer = AgentTracer(service_name="demo-agent", run_dir=run_dir)

        # Run multiple episodes under a benchmark
        with tracer.benchmark("demo_experiment"):
            for ep_num in range(3):
                with tracer.episode(f"episode_{ep_num}"):
                    for step_num in range(5):
                        tracer.log({"action": f"action_{step_num}", "reward": step_num * 0.1})

        tracer.shutdown()

        # Show what was created
        print("\nOutput structure:")
        for root, dirs, files in os.walk(run_dir):
            level = root.replace(run_dir, "").count(os.sep)
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")
            for f in sorted(files):
                print(f"{indent}  {f}")


if __name__ == "__main__":
    main()

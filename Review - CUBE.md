**Metareview:**  
CUBE proposes a unified Python standar for agentic benchmarks to solve the N×M integrationproblem across 600+ environments. It provides a commn interface for tasks, tools, and resources across 1 wrapped benchmarks (\~5,600 tasks), with a trajector judge that attributes failures to agent, tool, or bnchmark. Evaluation on 4 models finds \~82% of failurs trace to the LLM.

Addresses a genune fragmentation problem that blocks multi-environmet research. System design thoughtfully separates conerns (tasks, tools, configs, resources) with a threetier resource lifecycle. The decentralized registry ith automated LLM-free debug checks and GitHub Actios compliance enables trustworthy community contributons. Trajectory judge incorporates safeguards (evidece verification, K=3 voting, limited taxonomy) beyon typical LLM-as-judge approaches. Successfully integates diverse modalities enabling cross-benchmark anayses (e.g., tool-action limits as failure modes in MniWoB).

Wrapper faithfulness validaion is circular—the paper provides no direct parity hecks (original benchmark's reference agent in CUBE,or vice versa). The 82% blame rate from the authors'own judge doesn't establish wrapper validity. Judge alidation is incomplete: agreement metrics only cove SWE-bench; no reliability data for web or computer-se tasks where LLM judges fail most. Agreement measues consistency, not correctness—human validation on 0–100 samples is needed. Single-agent evaluation (Geny only) undermines generalizability; Appendix D nots different scaffolds can differ by tens of points. rimary motivation (RL post-training) is untested: ony closed API models evaluated, no open-weight modelsor actual training runs. Missing quantitative comparsons to Harbor, NeMo Gym, AgentBeats, OpenEnv on covrage, complexity, throughput.

Key Questions forAuthors: (1) Provide direct pass-rate comparisons beween CUBE-wrapped benchmarks and original implementaions. (2) Report human annotations on 50–100 failed pisodes with Cohen's kappa vs. judge predictions, coering all modalities. (3) Present results from at lest one additional agent scaffold to confirm robustnes of failure patterns. (4) Create comparison table wth existing frameworks on benchmark count, wrapping ffort, evaluation throughput. Points 1–2 are critica for publication; points 3–4 strengthen the contribuion substantially.

**Review \#1**

**Summary:**  
The paper proposes a framework to standardize multiple agent/benchmarks into a common python script format. The note that their framework supports a JSON-RPC interface for non-Python clients. They showcase their framework using a variety of agents/models and benchmarks.

**Contribution Type Check:** Yes  
**Strengths:**  
The paper works on an important problem and cites multiple benchmarks now standard in the literature. I think the idea of having a single python script to support non-python clients is ideal The work clear outlines the different stages in their pipeline, similar to other open source frameworks.

**Weaknesses:**  
My main concern is that I did not see results to validate the pipeline. The results shown with the baseline agent are quite bad. For example, Terminal Bench GPT 5.4 achieves 75% on the report [https://openai.com/index/introducing-gpt-5-4/](https://openai.com/index/introducing-gpt-5-4/) but with their agent 11.4% (Table 1). I did not see where the agents validated that their pipeline was able to achieve the same performance.

Similarly I think the paper would benefit from the authors explaining more why the agent abstraction they use is the right one. Do they see differences in how tool calls are processed? Do they see differences across modalities. I think the abstraction alone is not a key contribution given other frameworks such as harbor or nemorl, so if the authors think they have a better abstraction they should highlight this more. Similarly if the framework supports a JSON-RPC interface for non-Python clients. I think more time should be spent on this contribution which in my opinion is a differentiating factor.

**Ethical Concerns:** No or very minor ethics concerns only  
**Reproducibility:** Yes  
**Dataset Assessment:** NA \- no dataset included  
**Dataset Comments:**  
The code is included.

**Limitations:**  
Yes

**Questions:**

* What is the right abstraction for tool calling across multiple agents? For example MCP vs bash?  
* Do the authors see reward hacking as common failure?  
* Are the LLM judges for failure analysis calibrated per mode?

**Rating:** 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.  
**Confidence:** 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

**Answer:**

**Review \#2**

**Summary:**  
The paper implements a Python standard with task/benchmark/tool/config abstractions provisions docker/vm/live backends. It provides a composite benchmark config for mixing benchmarks, and an generated JSON-RPC server that exposes an MCP-compatible tool endpoint. They also implement a reference harness, ray based parallel runner, with an llm-based trajectory judge that classifies the primary failure cause to one of 10 categories grouped into agent / tool / benchmark buckets. They wrap 10 benchmarks: WorkArena, WebArena-Verified, MiniWoB, BrowseComp, OSWorld, Windows Agent Arena, SWE-bench Verified, SWE-bench Live, Terminal-Bench, DRBench with total of 5,642 tasks and add a github-Actions compliance pipeline. Four API models are evaluated on benchmarks, using one fixed tool stack per modality.

**Contribution Type Check:** Yes  
**Strengths:**  
This work is addressing a real problem. A standard that cover WebArena, OSWorld, and SWE-bench through a single API and already plugs into AgentBeats and NeMo Gym would be truly useful infrastructure for both evaluation and RL post training. The standard's design choices are solid. The llm-free debug suite and scripted debug agent, plus the CI compliance pipeline are very good practice for keeping a registry trustworthy. Limitations are clearly mentioned.

**Weaknesses:**

1. llm judge has no human ground-truth validation. The only validation is self-agreement among k runs. Would be great to see human annotation of a subset of episodes, maybe 50-100 across modalities.  
2. The judge could be biased towards claude models. Would be nice to re-run with GPT and compare results.  
3. Would be nice to have more models tested with coverage across different providers.

**Ethical Concerns:** No or very minor ethics concerns only  
**Reproducibility:** Yes  
**Dataset Assessment:** NA \- no dataset included  
**Limitations:**  
Yes. Authors mention streaming and multi-agent not yet supported.

**Questions:**

1. Can you show human (expert) labels for a stratified sample of 50 to 100 failed episodes?  
2. Does the agent/tool blame split change if the you switch the judge to a model other than claude?  
3. Can you add a table comparing CUBE against METR, Inspect AI, BrowserGym, AgentGym, and OpenEnv etc.?

**Rating:** 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.  
**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

**Answer:**

**Review \#3**

**Summary:**  
This paper presents CUBE (Common Unified Benchmark Environments) a python based standard that provides a single interface for tasks, tools and resource management across different agentic benchmarks. The paper's central goal is to remove the N x M integration cost that makes cross benchmark evaluation and RL training difficult. The authors also offer a reference harness, a generalist agent and trajectory judge which identifies whether the failure is due to the agent, the tool, or the benchmark itself. They integrate 10 exiting benchmarks (about 5600 tasks covering web, computer use, software engineering and terminal tasks) into a decentralized registry. They then evaluate 4 models on 8 of these benchmarks, keeping the tools same for each type. The main result is that about 82% of the failures are due to LLM or agent while the rest are split between tool stack and wrapper or benchmark issues. The authors argue that the low rate of benchmark side errors show that their adaptation of benchmarks is valid.

**Contribution Type Check:** Yes  
**Strengths:**

1. **The paper deals with a real and clearly defined problem.** The NxM integration cost across more than 600 benchmarks is big challenge for comparing benchmarks and multi-environment RL runs. The author explains it well using specific examples of setup methods that do not work together (WebArena shared server, SWEBench per task container, OSWorld's VM, WorkArena live ServiceNow instance).   
2. **The paper introduces clear and helpful ideas.** The standard is more than just a simple API wrapper. It separates settings (which can be saved and shared between processes). Tools are treated as fully replaceable parts, and three level resource lifecycle handles different setup methods through one interface (Section 3.1, 3.2). This shows careful design.  
3. **Using a registry and automated checks offers a reliable way to grow without central control.** The process of accepting third party benchmarks is clearly defined: a debug agent without LLM must get a perfect score on fixed debug tasks, checked by Github Actions and shown with status badge (Section 3.2, 5). This practical and verifiable method lets the collection grow beyond the original authors. It is probably paper's most important idea.  
4. **The trajectory judge is designed for reliability, not just evaluation based on prompts.** Beside using an LLM-as-a-judge, the system has safeguards: every quote is checked against the saved step file to catch hallucinated quotes, three reviewers use majorty voting, there is a limited 10 category taxonomy, and a clear "none" option stops making up reasons (Section 4.1, Appendix C).  
5. **The standard does not depends on specific tools and is easy to copy.** It works with AgentBeats and NemoGym using a connector of about 100 lines of code (Section 4). Code, debug tests and compliance checks are included (Appendix A). The single agent evaluation across benchmarks gives a ground for comparison which separate leaderboard miss. The trajectory analysis shows details that individual benchmarks misses, like revealing that majority of MiniWoB failures happen because of tool action limits rather than model ability (Section 6.3).

**Weaknesses:**

1. **Using only one agent scaffold makes it hard to compare results fairly and is mostly not usable for standard evaluation.** All evaluations is based on a single agent Genny. However, Appendix D notes that "two scaffolds calling the same model can differ by tens of points on the same benchmark." This means the paper recognizes that the result can change based on the scaffold but still reports only one. Other platforms like Harbor tests multiple agent scaffolds such as Claude Code, Codex, Gemini CLI, etc.  
2. **The paper claims that the wrappers are faithful, but doesn't actually measure this.** The main evidence is a low benchmark side blame rate that is measured using trajectory judge, which isn't validated against human judgement. The authors should either run the original benchmark's reference agent in their framework (CUBE), or run Genny agent in the original benchmark code and show parity. Harbor framework does an extensive parity in a similar way.  
3. **The trajectory judge hasn't been validated against human.** The judge is central to both the failure analysis and faithfulness argument, but Appendix E only reports agreement between judges, which shows consistency but not correctness. The cited works like MAST also includes human judgement with cohen kappa agreement between the human labels and the LLM.   
4. **The evidence for the judge reliability only covers SWE-tasks.** Tables 3 and 4 show agreement only for SWEBench Verified and SWEBench Live. There is no judge results for web or CUA tasks. The central claim of the paper is a unified framework for any type of environment, but lack of extensive evaluation of the judge on different types of environment.  
5. **The paper didn't evaluate their integrated environment with any RL training runs.** The main reason given for CUBE is to support multi-environment RL post-training, but the paper only evaluated 4 frontier closed API models and does not include any open weights models. Since RL needs models with updatable weights, none of the used models can be used for the stated purpose.  
6. **There is no quantative comparison to the previous frameworks.** Section 2 describes how CUBE differs from Harbor, NeMo Gym, AgentBeats, and OpenEnv but only in words. There are no direct comparison on benchmark coverage, training or evaluation throughput, scalability and wrapping effort, even though Harbor includes a much larger benchmark set.

**Ethical Concerns:** No or very minor ethics concerns only  
**Reproducibility:** Yes  
**Dataset Assessment:** NA \- no dataset included  
**Limitations:**  
The authors have a Limitation section 7 and are open about several boundaries of their work. Still the most important limitations are either not clearly stated or are missing.

1. The paper talks about judge accuracy but doesn't evaluate it completely. Section 7 says faithfulness is limited by judge accuracy and points to Appendix E, but the appendix only shows how much judges agree with each other, not how accurate are they compared to humans.   
2. Section 7 does not mention that the only proof for wrapper validation comes from author's own judge, using blame rate. Other comparisons which includes score matching with the original benchmark is not included in Appendix D.   
3. The paper does not report any RL post-training run or metrics quantifying the framework's usability. It says that it supports RL post-training, but only tests closed frontier models and does not show any training examples or metrics. Section 7 does not say that one of the main use case is not tested.

**Questions:**

1. **Score parity with original benchmarks.** Please provide a direct comparison of pass rates between the CUBE wrapper and the original benchmark. Can you implement the agent used by the benchmark integrated in the CUBE framework and compare the evaluation results with what is reported by the original benchmark?  
2. **Human validation of the judge.** Please report how closely the trajectory judge decisions match the human gold labels, even if you have 50-100 samples, it would be good enough. Showing that the judge's decision match human experts, similar to MAST work will prove the reliability of the judge.  
3. **Judge reliability coverage.** The judge agreement in Appendix E is only reported for SWEBench and SWEBench-Live. Please also report the agreement metrics for web and CUA benchmarks where the judge's interpretation is particularly important as LLM judges might be poor with multimodal evaluations.  
4. **Scaffold sensitivity.** As noted in Appendix D, two agent scaffolds can differ by tens of points. Please share results for atleast one more agent, such as second public agent on a subset of benchmark. If the rankings change a lot, it would help clarify how to interpret the results.  
5. **Open weight models and reinforcement learning claims.** The main reason for the framework given is RL post-training, but only closed frontier API models are tested. Please include results for atleast one open-weight model, and will it be possible to share the RL rollout throughput or training results using CUBE? Showing such comparison with Harbor which has integration with open source RL training frameworks will make the claim of RL post-training stronger.

**Rating:** 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.  
**Confidence:** 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

**Answer:**
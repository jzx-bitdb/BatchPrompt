# BatchPrompt

## Introduction
Large Language Models (LLMs) have recently demonstrated exceptional performance in various real-world data management tasks through in-context learning (ICL), which involves structuring prompts with task descriptions and several demonstrations. However, most LLMs are not free and charge based on the number of input tokens. Specifically, for data management tasks, there may be massive related questions, leading to high inference cost due to redundant prompt content (i.e., overlapping demonstrations and repeated task descriptions). In this paper, we investigate the idea of batch prompting in leveraging LLMs for data management, which leads to cost-effective LLMs by grouping questions and demonstrations to perform inferences in batches. Current studies on batch prompting are preliminary and mostly based on heuristics, making it difficult to generalize to various types of tasks and adapt to different grouping strategies. To address these challenges, in this work we first formalize the batch prompting problem in general setting. Then, we study the hardness of this problem and propose efficient algorithms for adaptive grouping. Finally, we conduct comprehensive experiments on 14 datasets. Extensive experimental results demonstrate that our solution consistently outperforms the state-of-the-art baselines while consuming lower cost.

## Quick Start

Some commands are in 'command.sh'.

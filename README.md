# BatchPrompt

## Introduction
To minimize LLM costs while maintaining the performance of LLMs, we explore factors that may affect the performance and formalize the Batch Prompting problem for the first time. Then, we study the hardness of this problem and propose a series of efficient exact and approximation algorithms with provable theoretical guarantees. Finally, we conduct comprehensive experiments to evaluate our method on twelve datasets. Extensive experimental results show that our proposed solution outperforms the state-of-the-art baselines while consuming fewer costs.

## Dependencies and Build
```
We use https://github.com/Garrafao/correlation_clustering  as our correlation clustering algorithm.
```

## Dataset
We use two tasks for evaluation, namely, Entity Resolution and Code Assertion Generation, which are in the directory 'data'.


## Quick Start

All commands are in 'command.sh'.

# Adversarial Training of Text Scoring Models

This repository contains code for training and evaluating robust text scoring models including retrievers, rerankers, and reward models, following our paper "Language Models Bleed the Same: A Holistic Study of Training for Adversarial Robustness Across Open-Domain Text Scoring Tasks".

This repository is currently a work in progress, please feel free to file an issue or contact us if you encounter any problems with the code.

We also make available some of our adversarially trained models trained with a combination of adversarial training methods as well as a fine-tuned Llama-3.1-8B-Instruct model trained with further RLHF using our adversarially trained reward model.

## 🤗 Hugging Face Models

We release adversarially trained text-scoring models (retrievers, rerankers, reward models) and an aligned Llama-3.1-8B-Instruct model.

| Category | Model | Link |
|---|---|---|
| **Aligned model** | `llama-3.1-8b-instruct-align-at-comb-high` | https://huggingface.co/manveertamber/llama-3.1-8b-instruct-align-at-comb-high |
| **Reward model** | `llama-3.1-8B-instruct-reward-at-combined-medium` | https://huggingface.co/manveertamber/llama-3.1-8B-instruct-reward-at-combined-medium |
| **Reward model** | `llama-3.1-8B-instruct-reward-at-combined-high` | https://huggingface.co/manveertamber/llama-3.1-8B-instruct-reward-at-combined-high |
| **Reward model** | `llama-3.2-3B-instruct-reward-at-combined-medium` | https://huggingface.co/manveertamber/llama-3.2-3B-instruct-reward-at-combined-medium |
| **Reward model** | `llama-3.2-3B-instruct-reward-at-combined-high` | https://huggingface.co/manveertamber/llama-3.2-3B-instruct-reward-at-combined-high |
| **Reranker** | `qwen3-0.6B-reranker-at-combined-medium` | https://huggingface.co/manveertamber/qwen3-0.6B-reranker-at-combined-medium |
| **Retriever** | `bert-base-retriever-at-combined-medium` | https://huggingface.co/manveertamber/bert-base-retriever-at-combined-medium |

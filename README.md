# Awesome Post-Training for MLLMs

![Post-training framework for MLLMs](assets/framework.png)

A curated paper collection for **post-training Multimodal Large Language Models (MLLMs)**, organized to match a survey-style taxonomy: instruction tuning, data construction, preference alignment, reinforcement learning, hallucination mitigation, reasoning, video/omni-modal adaptation, agents/grounding, domain post-training, and evaluation.

The collection focuses on papers that are directly useful for understanding how MLLMs are adapted after pre-training. Each entry tries to include:

- **Web**: project page, blog, model card, or collection page when available.
- **Paper**: arXiv, proceedings, or technical report.
- **Code**: official repository, training code, evaluation code, or dataset repository when available.

`-` means I did not find a reliable official code/project link.

## Contents

- [Surveys and Roadmaps](#surveys-and-roadmaps)
- [Post-Training Foundations](#post-training-foundations)
- [General MLLMs and Visual Instruction Tuning](#general-mllms-and-visual-instruction-tuning)
- [Instruction Data, Data Engines, and Synthetic Supervision](#instruction-data-data-engines-and-synthetic-supervision)
- [Preference Alignment, RLHF, DPO, and Reward Models](#preference-alignment-rlhf-dpo-and-reward-models)
- [Reinforcement Learning for Multimodal Reasoning](#reinforcement-learning-for-multimodal-reasoning)
- [Hallucination, Faithfulness, and Safety](#hallucination-faithfulness-and-safety)
- [Multimodal Reasoning and Chain-of-Thought](#multimodal-reasoning-and-chain-of-thought)
- [Video, Audio, and Omni-Modal Post-Training](#video-audio-and-omni-modal-post-training)
- [Grounding, GUI Agents, and Tool Use](#grounding-gui-agents-and-tool-use)
- [Domain-Specific Post-Training](#domain-specific-post-training)
- [Evaluation and Benchmarks](#evaluation-and-benchmarks)
- [Related Awesome Lists](#related-awesome-lists)

## Surveys and Roadmaps

| Year | Title | Web | Paper | Code/List |
|---|---|---|---|---|
| 2025 | Aligning Multimodal LLM with Human Preference: A Survey | [HF](https://huggingface.co/papers/2503.14504) | [arXiv](https://arxiv.org/abs/2503.14504) | - |
| 2025 | Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal Large Language Models | [arXiv](https://arxiv.org/abs/2504.21277) | [arXiv](https://arxiv.org/abs/2504.21277) | - |
| 2025 | Self-Improvement in Multimodal Large Language Models: A Survey | [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.105/) | [arXiv](https://arxiv.org/abs/2510.02665) | - |
| 2025 | Towards LLM-Centric Multimodal Fusion: A Survey on Integration Strategies and Techniques | [HF](https://huggingface.co/papers/2506.04788) | [arXiv](https://arxiv.org/abs/2506.04788) | - |
| 2025 | On Domain-Specific Post-Training for Multimodal Large Language Models | [arXiv](https://arxiv.org/abs/2411.19930) | [arXiv](https://arxiv.org/abs/2411.19930) | - |
| 2024 | MM-LLMs: Recent Advances in MultiModal Large Language Models | [Project](https://mm-llms.github.io/) | [arXiv](https://arxiv.org/abs/2401.13601) | [GitHub](https://github.com/MM-LLMs/MM-LLMs.github.io) |
| 2024 | A Survey on Evaluation of Multimodal Large Language Models | [arXiv](https://arxiv.org/abs/2408.15769) | [arXiv](https://arxiv.org/abs/2408.15769) | - |
| 2024 | MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs | [List](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Benchmarks) | [arXiv](https://arxiv.org/abs/2411.15296) | [GitHub](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Benchmarks) |
| 2023 | A Survey on Multimodal Large Language Models | [NSR](https://academic.oup.com/nsr/article/11/12/nwae403/7909600) | [arXiv](https://arxiv.org/abs/2306.13549) | [GitHub](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) |
| 2023 | The Revolution of Multimodal Large Language Models: A Survey | [ACL](https://aclanthology.org/2024.findings-acl.807/) | [PDF](https://aclanthology.org/2024.findings-acl.807.pdf) | - |

## Post-Training Foundations

These are not all multimodal, but they are the backbone methods repeatedly reused in MLLM post-training.

| Year | Title | Web | Paper | Code |
|---|---|---|---|---|
| 2025 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | [Project](https://github.com/deepseek-ai/DeepSeek-R1) | [arXiv](https://arxiv.org/abs/2501.12948) | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) |
| 2024 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | [Project](https://github.com/deepseek-ai/DeepSeek-Math) | [arXiv](https://arxiv.org/abs/2402.03300) | [GitHub](https://github.com/deepseek-ai/DeepSeek-Math) |
| 2024 | Kahneman-Tversky Optimization: Human-Aligned LLMs | [Project](https://github.com/ContextualAI/HALOs) | [arXiv](https://arxiv.org/abs/2402.01306) | [GitHub](https://github.com/ContextualAI/HALOs) |
| 2024 | ORPO: Monolithic Preference Optimization without Reference Model | [Project](https://github.com/xfactlab/orpo) | [arXiv](https://arxiv.org/abs/2403.07691) | [GitHub](https://github.com/xfactlab/orpo) |
| 2024 | SimPO: Simple Preference Optimization with a Reference-Free Reward | [Project](https://github.com/princeton-nlp/SimPO) | [arXiv](https://arxiv.org/abs/2405.14734) | [GitHub](https://github.com/princeton-nlp/SimPO) |
| 2023 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | [Project](https://github.com/eric-mitchell/direct-preference-optimization) | [arXiv](https://arxiv.org/abs/2305.18290) | [GitHub](https://github.com/eric-mitchell/direct-preference-optimization) |
| 2023 | RRHF: Rank Responses to Align Language Models with Human Feedback | [Project](https://github.com/GanjinZero/RRHF) | [arXiv](https://arxiv.org/abs/2304.05302) | [GitHub](https://github.com/GanjinZero/RRHF) |
| 2023 | Training Socially Aligned Language Models on Simulated Social Interactions | - | [arXiv](https://arxiv.org/abs/2305.16960) | - |
| 2023 | RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback | - | [arXiv](https://arxiv.org/abs/2309.00267) | - |
| 2023 | Alpaca: A Strong, Replicable Instruction-Following Model | [Blog](https://crfm.stanford.edu/2023/03/13/alpaca.html) | - | [GitHub](https://github.com/tatsu-lab/stanford_alpaca) |
| 2022 | Training Language Models to Follow Instructions with Human Feedback | [OpenAI](https://openai.com/index/instruction-following/) | [arXiv](https://arxiv.org/abs/2203.02155) | - |
| 2022 | Constitutional AI: Harmlessness from AI Feedback | - | [arXiv](https://arxiv.org/abs/2212.08073) | - |
| 2022 | Self-Instruct: Aligning Language Models with Self-Generated Instructions | [Project](https://self-instruct.github.io/) | [arXiv](https://arxiv.org/abs/2212.10560) | [GitHub](https://github.com/yizhongw/self-instruct) |
| 2022 | Scaling Instruction-Finetuned Language Models | - | [arXiv](https://arxiv.org/abs/2210.11416) | - |
| 2021 | Finetuned Language Models Are Zero-Shot Learners | - | [arXiv](https://arxiv.org/abs/2109.01652) | - |

## General MLLMs and Visual Instruction Tuning

| Year | Title | Web | Paper | Code |
|---|---|---|---|---|
| 2025 | Qwen2.5-VL Technical Report | [Blog](https://qwenlm.github.io/blog/qwen2.5-vl/) | [arXiv](https://arxiv.org/abs/2502.13923) | [GitHub](https://github.com/QwenLM/Qwen2.5-VL) |
| 2025 | InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency | [Project](https://internvl.github.io/) | [arXiv](https://arxiv.org/abs/2508.18265) | [GitHub](https://github.com/OpenGVLab/InternVL) |
| 2025 | GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning | [Project](https://github.com/THUDM/GLM-4.1V-Thinking) | [arXiv](https://arxiv.org/abs/2507.01006) | [GitHub](https://github.com/THUDM/GLM-4.1V-Thinking) |
| 2024 | LLaVA-OneVision: Easy Visual Task Transfer | [Project](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/) | [arXiv](https://arxiv.org/abs/2408.03326) | [GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT) |
| 2024 | Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs | [Project](https://cambrian-mllm.github.io/) | [arXiv](https://arxiv.org/abs/2406.16860) | [GitHub](https://github.com/cambrian-mllm/cambrian) |
| 2024 | Chameleon: Mixed-Modal Early-Fusion Foundation Models | [Meta](https://ai.meta.com/research/publications/chameleon-mixed-modal-early-fusion-foundation-models/) | [arXiv](https://arxiv.org/abs/2405.09818) | - |
| 2024 | How Far Are We to GPT-4V? Closing the Gap with InternVL 1.5 | [Project](https://internvl.github.io/blog/2024-04-30-InternVL-1.5/) | [arXiv](https://arxiv.org/abs/2404.16821) | [GitHub](https://github.com/OpenGVLab/InternVL) |
| 2024 | MM1: Methods, Analysis and Insights from Multimodal LLM Pre-training | [Apple](https://machinelearning.apple.com/research/mm1) | [arXiv](https://arxiv.org/abs/2403.09611) | - |
| 2024 | LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge | [Project](https://llava-vl.github.io/blog/2024-01-30-llava-next/) | - | [GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT) |
| 2024 | Unified-IO 2: Scaling Autoregressive Multimodal Models with Vision, Language, Audio, and Action | [Project](https://unified-io-2.allenai.org/) | [arXiv](https://arxiv.org/abs/2312.17172) | [GitHub](https://github.com/allenai/unified-io-2) |
| 2024 | Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution | [Blog](https://qwenlm.github.io/blog/qwen2-vl/) | [arXiv](https://arxiv.org/abs/2409.12191) | [GitHub](https://github.com/QwenLM/Qwen2-VL) |
| 2024 | Llama 3.2 Vision | [Meta](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) | [Model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md) | [GitHub](https://github.com/meta-llama/llama-models) |
| 2024 | Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation | [Project](https://github.com/deepseek-ai/Janus) | [arXiv](https://arxiv.org/abs/2410.13848) | [GitHub](https://github.com/deepseek-ai/Janus) |
| 2024 | Emu3: Next-Token Prediction is All You Need | [Project](https://baaivision.github.io/emu3/) | [arXiv](https://arxiv.org/abs/2409.18869) | [GitHub](https://github.com/baaivision/Emu3) |
| 2024 | Show-o: One Single Transformer to Unify Multimodal Understanding and Generation | [Project](https://showlab.github.io/Show-o/) | [arXiv](https://arxiv.org/abs/2408.12528) | [GitHub](https://github.com/showlab/Show-o) |
| 2023 | LLaVA: Visual Instruction Tuning | [Project](https://llava-vl.github.io/) | [arXiv](https://arxiv.org/abs/2304.08485) | [GitHub](https://github.com/haotian-liu/LLaVA) |
| 2023 | Improved Baselines with Visual Instruction Tuning | [Project](https://llava-vl.github.io/) | [arXiv](https://arxiv.org/abs/2310.03744) | [GitHub](https://github.com/haotian-liu/LLaVA) |
| 2023 | MiniGPT-4: Enhancing Vision-Language Understanding with Advanced LLMs | [Project](https://minigpt-4.github.io/) | [arXiv](https://arxiv.org/abs/2304.10592) | [GitHub](https://github.com/Vision-CAIR/MiniGPT-4) |
| 2023 | InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning | [Project](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | [arXiv](https://arxiv.org/abs/2305.06500) | [GitHub](https://github.com/salesforce/LAVIS) |
| 2023 | mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality | [Project](https://mplug-owl.github.io/) | [arXiv](https://arxiv.org/abs/2304.14178) | [GitHub](https://github.com/X-PLUG/mPLUG-Owl) |
| 2023 | Otter: A Multi-Modal Model with In-Context Instruction Tuning | [Project](https://luodian.github.io/Otter/) | [arXiv](https://arxiv.org/abs/2305.03726) | [GitHub](https://github.com/Luodian/Otter) |
| 2023 | MultiModal-GPT: A Vision and Language Model for Dialogue with Humans | [Project](https://openmmlab.github.io/Multimodal-GPT/) | [arXiv](https://arxiv.org/abs/2305.04790) | [GitHub](https://github.com/OpenGVLab/MultiModal-GPT) |
| 2023 | LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model | [Project](https://github.com/OpenGVLab/LLaMA-Adapter) | [arXiv](https://arxiv.org/abs/2304.15010) | [GitHub](https://github.com/OpenGVLab/LLaMA-Adapter) |
| 2023 | LaVIN: Large Vision-Language Instructed Model | [Project](https://luogen1996.github.io/lavin/) | [arXiv](https://arxiv.org/abs/2305.15023) | [GitHub](https://github.com/luogen1996/LaVIN) |
| 2023 | Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic | [Project](https://shikras.github.io/) | [arXiv](https://arxiv.org/abs/2306.15195) | [GitHub](https://github.com/shikras/shikra) |
| 2023 | Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond | [Blog](https://qwenlm.github.io/blog/qwen-vl/) | [arXiv](https://arxiv.org/abs/2308.12966) | [GitHub](https://github.com/QwenLM/Qwen-VL) |
| 2023 | CogVLM: Visual Expert for Pretrained Language Models | [Project](https://github.com/THUDM/CogVLM) | [arXiv](https://arxiv.org/abs/2311.03079) | [GitHub](https://github.com/THUDM/CogVLM) |
| 2023 | VILA: On Pre-training for Visual Language Models | [Project](https://hanlab.mit.edu/projects/vila) | [arXiv](https://arxiv.org/abs/2312.07533) | [GitHub](https://github.com/NVlabs/VILA) |
| 2023 | Emu2: Generative Multimodal Models are In-Context Learners | [Project](https://baaivision.github.io/emu2/) | [arXiv](https://arxiv.org/abs/2312.13286) | [GitHub](https://github.com/baaivision/Emu) |
| 2023 | Fuyu-8B: A Multimodal Architecture for AI Agents | [Blog](https://www.adept.ai/blog/fuyu-8b) | - | [HF](https://huggingface.co/adept/fuyu-8b) |
| 2023 | Kosmos-1: Language Is Not All You Need | [Project](https://github.com/microsoft/unilm/tree/master/kosmos-1) | [arXiv](https://arxiv.org/abs/2302.14045) | [GitHub](https://github.com/microsoft/unilm) |
| 2023 | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs | [Project](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | [arXiv](https://arxiv.org/abs/2301.12597) | [GitHub](https://github.com/salesforce/LAVIS) |
| 2022 | Flamingo: A Visual Language Model for Few-Shot Learning | [DeepMind](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model) | [arXiv](https://arxiv.org/abs/2204.14198) | - |
| 2022 | PaLI: A Jointly-Scaled Multilingual Language-Image Model | [Google](https://ai.googleblog.com/2022/09/pali-scaling-language-image-learning-in.html) | [arXiv](https://arxiv.org/abs/2209.06794) | - |

## Instruction Data, Data Engines, and Synthetic Supervision

| Year | Title / Dataset | Web | Paper | Code / Data |
|---|---|---|---|---|
| 2025 | DenseWorld-1M: Detailed Dense Grounded Caption in the Real World | [Project](https://denseworld.github.io/) | [arXiv](https://arxiv.org/abs/2506.24102) | [GitHub](https://github.com/lxtGH/DenseWorld-1M) |
| 2023 | MLLM-DataEngine: Closing the Loop of Instruction Tuning Data Generation | [Project](https://juliozhao97.github.io/MLLM-DataEngine/) | [arXiv](https://arxiv.org/abs/2308.13566) | [GitHub](https://github.com/opendatalab/MLLM-DataEngine) |
| 2024 | Cambrian-10M: A Fully Open Instruction Tuning Corpus | [Project](https://cambrian-mllm.github.io/) | [arXiv](https://arxiv.org/abs/2406.16860) | [GitHub](https://github.com/cambrian-mllm/cambrian) |
| 2024 | ALLaVA: Harnessing GPT4V-Synthesized Data for A Lite Vision-Language Model | [Project](https://allava.freedomai.cn/) | [arXiv](https://arxiv.org/abs/2402.11684) | [GitHub](https://github.com/FreedomIntelligence/ALLaVA) |
| 2024 | Vision-Flan: Scaling Human-Labeled Tasks in Visual Instruction Tuning | [Project](https://vision-flan.github.io/) | [arXiv](https://arxiv.org/abs/2402.11690) | - |
| 2024 | LLaVA-NeXT-Data: A Large-Scale Visual Instruction Dataset | [HF](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data) | - | [HF](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data) |
| 2023 | ShareGPT4V: Improving Large Multi-Modal Models with Better Captions | [Project](https://sharegpt4v.github.io/) | [arXiv](https://arxiv.org/abs/2311.12793) | [GitHub](https://github.com/ShareGPT4Omni/ShareGPT4V) |
| 2023 | LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset | [Project](https://openlamm.github.io/) | [arXiv](https://arxiv.org/abs/2306.06687) | [GitHub](https://github.com/OpenLAMM/LAMM) |
| 2023 | M3IT: A Large-Scale Multi-Modal Instruction Tuning Dataset | [Project](https://m3-it.github.io/) | [arXiv](https://arxiv.org/abs/2306.04387) | [GitHub](https://github.com/DAMO-NLP-SG/Multimodal-Instruction-Tuning) |
| 2023 | MIMIC-IT: Multi-Modal In-Context Instruction Tuning | [Project](https://luodian.github.io/Otter/) | [arXiv](https://arxiv.org/abs/2306.05425) | [GitHub](https://github.com/Luodian/Otter/tree/main/mimic-it) |
| 2023 | LLaVA-Instruct-150K | [Project](https://llava-vl.github.io/) | [arXiv](https://arxiv.org/abs/2304.08485) | [Data](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) |
| 2023 | LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding | [Project](https://llavar.github.io/) | [arXiv](https://arxiv.org/abs/2306.17107) | [GitHub](https://github.com/SALT-NLP/LLaVAR) |
| 2023 | SVIT: Scaling Up Visual Instruction Tuning | [Project](https://svit-vl.github.io/) | [arXiv](https://arxiv.org/abs/2307.04087) | [GitHub](https://github.com/BAAI-DCAI/Visual-Instruction-Tuning) |
| 2022 | MultiInstruct: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning | [Project](https://multiinstruct.github.io/) | [arXiv](https://arxiv.org/abs/2212.10773) | [GitHub](https://github.com/VT-NLP/MultiInstruct) |

## Preference Alignment, RLHF, DPO, and Reward Models

| Year | Title | Web | Paper | Code / Data |
|---|---|---|---|---|
| 2025 | Safe RLHF-V: Safe Reinforcement Learning from Human Feedback in MLLMs | [Project](https://saferlhf-v.github.io/) | [arXiv](https://arxiv.org/abs/2503.17682) | [GitHub](https://github.com/SafeRLHF-V) |
| 2025 | R1-Reward: Training Multimodal Reward Model Through Stable Reinforcement Learning | [Project](https://github.com/yfzhang114/r1_reward) | [arXiv](https://arxiv.org/abs/2505.02835) | [GitHub](https://github.com/yfzhang114/r1_reward) |
| 2025 | LPOI: Listwise Preference Optimization for Vision-Language Models | [Project](https://github.com/fatemehpesaran310/lpoi) | [arXiv](https://arxiv.org/abs/2505.21061) | [GitHub](https://github.com/fatemehpesaran310/lpoi) |
| 2025 | M3PO: Multimodal-Model-Guided Preference Optimization for Visual Instruction Following | - | [arXiv](https://arxiv.org/abs/2508.12458) | - |
| 2026 | DA-DPO: Cost-efficient Difficulty-aware Preference Optimization for Reducing MLLM Hallucinations | [HF](https://huggingface.co/papers/2601.00623) | [arXiv](https://arxiv.org/abs/2601.00623) | - |
| 2025 | Direct Preference Optimization of Video Large Multimodal Models from Language Model Reward | - | [ACL](https://aclanthology.org/2025.naacl-long.30/) | [GitHub](https://github.com/RifleZhang/LLaVA-Hound-DPO) |
| 2024 | VLFeedback: A Large-Scale AI Feedback Dataset for LVLM Alignment | [Project](https://vlf-silkie.github.io/) | [arXiv](https://arxiv.org/abs/2410.09421) | [GitHub](https://github.com/vlf-silkie/VLFeedback) |
| 2024 | RLAIF-V: Aligning MLLMs through Open-Source AI Feedback for Super GPT-4V Trustworthiness | [HF](https://huggingface.co/openbmb/RLHF-V) | [arXiv](https://arxiv.org/abs/2405.17220) | [GitHub](https://github.com/RLHF-V/RLAIF-V) |
| 2024 | V-DPO: Mitigating Hallucination in LVLMs via Vision-Guided DPO | - | [arXiv](https://arxiv.org/abs/2411.02712) | - |
| 2024 | HSA-DPO: Detecting and Mitigating Hallucination via Fine-Grained AI Feedback | [Project](https://github.com/Mr-Loevan/HSA-DPO) | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34744) | [GitHub](https://github.com/Mr-Loevan/HSA-DPO) |
| 2024 | Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware DPO | [Project](https://opendatalab.github.io/HA-DPO/) | [arXiv](https://arxiv.org/abs/2311.16839) | [GitHub](https://github.com/opendatalab/HA-DPO) |
| 2023 | LLaVA-RLHF: Aligning Large Multimodal Models with Factually Augmented RLHF | [Project](https://llava-rlhf.github.io/) | [arXiv](https://arxiv.org/abs/2309.14525) | [GitHub](https://github.com/llava-rlhf/LLaVA-RLHF) |
| 2023 | RLHF-V: Trustworthy MLLMs via Fine-grained Correctional Human Feedback | [HF](https://huggingface.co/openbmb/RLHF-V) | [arXiv](https://arxiv.org/abs/2312.00849) | [GitHub](https://github.com/RLHF-V/RLHF-V) |
| 2023 | Silkie: Preference Distillation for Large Visual Language Models | [Project](https://vlf-silkie.github.io/) | [arXiv](https://arxiv.org/abs/2312.10665) | [GitHub](https://github.com/vlf-silkie/VLFeedback) |

## Reinforcement Learning for Multimodal Reasoning

| Year | Title | Web | Paper | Code |
|---|---|---|---|---|
| 2025 | GLM-4.1V-Thinking: Scalable Reinforcement Learning for Multimodal Reasoning | [Project](https://github.com/THUDM/GLM-4.1V-Thinking) | [arXiv](https://arxiv.org/abs/2507.01006) | [GitHub](https://github.com/THUDM/GLM-4.1V-Thinking) |
| 2025 | Perception-Aware Policy Optimization for Multimodal Reasoning | [HF](https://huggingface.co/papers/2507.06448) | [arXiv](https://arxiv.org/abs/2507.06448) | [GitHub](https://github.com/PAPO-Galaxy/PAPO) |
| 2025 | GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning | [HF](https://huggingface.co/papers/2506.16141) | [arXiv](https://arxiv.org/abs/2506.16141) | [GitHub](https://github.com/TencentARC/GRPO-CARE) |
| 2025 | ReVisual-R1: From Optimized Cold Start to Staged Reinforcement Learning | [Project](https://github.com/CSfufu/Revisual-R1) | [arXiv](https://arxiv.org/abs/2506.04207) | [GitHub](https://github.com/CSfufu/Revisual-R1) |
| 2025 | R1-ShareVL: Incentivizing Reasoning Capability via Share-GRPO | [HF](https://huggingface.co/papers/2505.16673) | [arXiv](https://arxiv.org/abs/2505.16673) | [GitHub](https://github.com/HJYao00/R1-ShareVL) |
| 2025 | Vision-R1: Incentivizing Reasoning Capability in MLLMs | [Project](https://github.com/Osilly/Vision-R1) | [arXiv](https://arxiv.org/abs/2503.06749) | [GitHub](https://github.com/Osilly/Vision-R1) |
| 2025 | R1-VL: Learning to Reason with MLLMs via Step-wise GRPO | [Project](https://github.com/jingyi0000/R1-VL) | [arXiv](https://arxiv.org/abs/2503.12937) | [GitHub](https://github.com/jingyi0000/R1-VL) |
| 2025 | Visionary-R1: Mitigating Shortcuts in Visual Reasoning with RL | [Project](https://github.com/maifoundations/Visionary-R1) | [arXiv](https://arxiv.org/abs/2505.14677) | [GitHub](https://github.com/maifoundations/Visionary-R1) |
| 2025 | MMSearch-R1: Incentivizing LMMs to Search | [Project](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1) | [arXiv](https://arxiv.org/abs/2506.20670) | [GitHub](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1) |
| 2025 | Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning | [Project](https://github.com/egolife-ai/Ego-R1) | [arXiv](https://arxiv.org/abs/2506.13654) | [GitHub](https://github.com/egolife-ai/Ego-R1) |

## Hallucination, Faithfulness, and Safety

| Year | Title | Web | Paper | Code / Data |
|---|---|---|---|---|
| 2025 | Safe RLHF-V: Safe RLHF in Multimodal Large Language Models | [Project](https://saferlhf-v.github.io/) | [arXiv](https://arxiv.org/abs/2503.17682) | [GitHub](https://github.com/SafeRLHF-V) |
| 2024 | Look Twice Before You Answer: Memory-Space Visual Retracing | [Project](https://github.com/1zhou-Wang/MemVR) | [arXiv](https://arxiv.org/abs/2410.03577) | [GitHub](https://github.com/1zhou-Wang/MemVR) |
| 2024 | HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding | [Project](https://github.com/BillChan226/HALC) | [arXiv](https://arxiv.org/abs/2403.00425) | [GitHub](https://github.com/BillChan226/HALC) |
| 2024 | IBD: Alleviating Hallucinations via Image-Biased Decoding | - | [arXiv](https://arxiv.org/abs/2402.18476) | - |
| 2024 | Less is More: Mitigating Multimodal Hallucination from an EOS Decision Perspective | [Project](https://github.com/yuezih/less-is-more) | [arXiv](https://arxiv.org/abs/2402.14545) | [GitHub](https://github.com/yuezih/less-is-more) |
| 2024 | Visually Dehallucinative Instruction Generation: Know What You Don't Know | [Project](https://github.com/ncsoft/idk) | [arXiv](https://arxiv.org/abs/2402.09717) | [GitHub](https://github.com/ncsoft/idk) |
| 2023 | POPE: Evaluating Object Hallucination in Large Vision-Language Models | [Project](https://github.com/RUCAIBox/POPE) | [arXiv](https://arxiv.org/abs/2305.10355) | [GitHub](https://github.com/RUCAIBox/POPE) |
| 2023 | HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion | [Project](https://github.com/tianyi-lab/HallusionBench) | [arXiv](https://arxiv.org/abs/2310.14566) | [GitHub](https://github.com/tianyi-lab/HallusionBench) |
| 2023 | Woodpecker: Hallucination Correction for Multimodal Large Language Models | [Project](https://woodpecker-mllm.github.io/) | [arXiv](https://arxiv.org/abs/2310.16045) | [GitHub](https://github.com/BradyFU/Woodpecker) |
| 2023 | OPERA: Alleviating Hallucination in MLLMs via Over-Trust Penalty and Retrospection-Allocation | [Project](https://github.com/shikiw/OPERA) | [arXiv](https://arxiv.org/abs/2311.17911) | [GitHub](https://github.com/shikiw/OPERA) |
| 2023 | VCD: Mitigating Object Hallucinations via Visual Contrastive Decoding | [Project](https://github.com/DAMO-NLP-SG/VCD) | [arXiv](https://arxiv.org/abs/2311.16922) | [GitHub](https://github.com/DAMO-NLP-SG/VCD) |

## Multimodal Reasoning and Chain-of-Thought

| Year | Title | Web | Paper | Code / Data |
|---|---|---|---|---|
| 2025 | Thyme: Think Beyond Images | [Project](https://thyme-vl.github.io/) | [arXiv](https://arxiv.org/abs/2508.11630) | [GitHub](https://github.com/yfzhang114/Thyme) |
| 2024 | Insight-V: Exploring Long-Chain Visual Reasoning with MLLMs | [Project](https://github.com/dongyh20/Insight-V) | [arXiv](https://arxiv.org/abs/2411.14432) | [GitHub](https://github.com/dongyh20/Insight-V) |
| 2024 | Cantor: Inspiring Multimodal Chain-of-Thought of MLLM | [Project](https://ggg0919.github.io/cantor/) | [arXiv](https://arxiv.org/abs/2404.16033) | [GitHub](https://github.com/ggg0919/cantor) |
| 2024 | Visual CoT: Unleashing Chain-of-Thought Reasoning in MLLMs | [Project](https://github.com/deepcs233/Visual-CoT) | [arXiv](https://arxiv.org/abs/2403.16999) | [GitHub](https://github.com/deepcs233/Visual-CoT) |
| 2024 | MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems? | [Project](https://mathverse-cuhk.github.io/) | [arXiv](https://arxiv.org/abs/2403.14624) | [GitHub](https://github.com/ZrrSkywalker/MathVerse) |
| 2024 | Math-Vision: Measuring Multimodal Mathematical Reasoning | [Project](https://mathvision-cuhk.github.io/) | [arXiv](https://arxiv.org/abs/2402.14804) | [GitHub](https://github.com/mathvision-cuhk/MathVision) |
| 2024 | CharXiv: Charting Gaps in Realistic Chart Understanding | [Project](https://charxiv.github.io/) | [arXiv](https://arxiv.org/abs/2406.18521) | [HF](https://huggingface.co/datasets/princeton-nlp/CharXiv) |
| 2024 | M3CoT: Multi-Domain Multi-step Multi-modal Chain-of-Thought | [Project](https://lightchen233.github.io/m3cot.github.io/) | [arXiv](https://arxiv.org/abs/2405.16473) | [GitHub](https://github.com/LightChen233/M3CoT) |
| 2023 | Multimodal Chain-of-Thought Reasoning in Language Models | [Project](https://github.com/amazon-science/mm-cot) | [arXiv](https://arxiv.org/abs/2302.00923) | [GitHub](https://github.com/amazon-science/mm-cot) |
| 2023 | Chameleon: Plug-and-Play Compositional Reasoning with LLMs | [Project](https://chameleon-llm.github.io/) | [arXiv](https://arxiv.org/abs/2304.09842) | [GitHub](https://github.com/lupantech/chameleon-llm) |
| 2023 | Visual Chain of Thought: Bridging Logical Gaps with Multimodal Infillings | [Project](https://vcot-vqa.github.io/) | [arXiv](https://arxiv.org/abs/2305.02317) | [GitHub](https://github.com/dannyrose30/VCOT) |
| 2023 | Caption Anything: Interactive Image Description with Multimodal Controls | [Project](https://ttengwang.github.io/Caption-Anything/) | [arXiv](https://arxiv.org/abs/2305.02677) | [GitHub](https://github.com/ttengwang/Caption-Anything) |
| 2023 | T-SciQ: Teaching Multimodal Chain-of-Thought for Science QA | - | [arXiv](https://arxiv.org/abs/2305.03453) | - |
| 2023 | DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning | [Project](https://github.com/SooLab/DDCOT) | [arXiv](https://arxiv.org/abs/2310.16436) | [GitHub](https://github.com/SooLab/DDCOT) |
| 2023 | Compositional Chain-of-Thought Prompting for Large Multimodal Models | [Project](https://github.com/chancharikmitra/CCoT) | [arXiv](https://arxiv.org/abs/2311.17076) | [GitHub](https://github.com/chancharikmitra/CCoT) |
| 2023 | MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts | [Project](https://mathvista.github.io/) | [arXiv](https://arxiv.org/abs/2310.02255) | [GitHub](https://github.com/lmms-lab/MathVista) |

## Video, Audio, and Omni-Modal Post-Training

| Year | Title | Web | Paper | Code |
|---|---|---|---|---|
| 2025 | Qwen3-Omni Technical Report | [Blog](https://qwenlm.github.io/blog/qwen3-omni/) | [arXiv](https://arxiv.org/abs/2509.17765) | [GitHub](https://github.com/QwenLM/Qwen3-Omni) |
| 2025 | BLIP3-o: Fully Open Unified Multimodal Models | [Project](https://github.com/JiuhaiChen/BLIP3o) | [arXiv](https://arxiv.org/abs/2505.09568) | [GitHub](https://github.com/JiuhaiChen/BLIP3o) |
| 2025 | MiniCPM-o: A GPT-4o Level MLLM for Vision, Speech, and Multimodal Interaction | [Project](https://openbmb.vercel.app/minicpm-o) | [Tech report](https://github.com/OpenBMB/MiniCPM-o) | [GitHub](https://github.com/OpenBMB/MiniCPM-o) |
| 2024 | VITA: Towards Open-Source Interactive Omni Multimodal LLM | [Project](https://vita-home.github.io/) | [arXiv](https://arxiv.org/abs/2408.05211) | [GitHub](https://github.com/VITA-MLLM/VITA) |
| 2024 | Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming | [Project](https://github.com/gpt-omni/mini-omni) | [arXiv](https://arxiv.org/abs/2408.16725) | [GitHub](https://github.com/gpt-omni/mini-omni) |
| 2024 | LLaMA-Omni: Seamless Speech Interaction with LLMs | [Project](https://github.com/ictnlp/LLaMA-Omni) | [arXiv](https://arxiv.org/abs/2409.06666) | [GitHub](https://github.com/ictnlp/LLaMA-Omni) |
| 2024 | Qwen2-Audio Technical Report | [Blog](https://qwenlm.github.io/blog/qwen2-audio/) | [arXiv](https://arxiv.org/abs/2407.10759) | [GitHub](https://github.com/QwenLM/Qwen2-Audio) |
| 2024 | AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling | [Project](https://junzhan2000.github.io/AnyGPT.github.io/) | [arXiv](https://arxiv.org/abs/2402.12226) | [GitHub](https://github.com/OpenMOSS/AnyGPT) |
| 2024 | LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding | [Project](https://vision-cair.github.io/LongVU/) | [arXiv](https://arxiv.org/abs/2410.17434) | [GitHub](https://github.com/Vision-CAIR/LongVU) |
| 2024 | LongVA: Long Context Transfer from Language to Vision | [Project](https://lmms-lab.github.io/LongVA/) | [arXiv](https://arxiv.org/abs/2406.16852) | [GitHub](https://github.com/lmms-lab/LongVA) |
| 2024 | MiniGPT4-Video: Interleaved Visual-Textual Tokens for Video Understanding | [Project](https://vision-cair.github.io/MiniGPT4-video/) | [arXiv](https://arxiv.org/abs/2404.03413) | [GitHub](https://github.com/Vision-CAIR/MiniGPT4-video) |
| 2023 | Video-LLaVA: Learning United Visual Representation by Alignment Before Projection | [Project](https://github.com/PKU-YuanGroup/Video-LLaVA) | [arXiv](https://arxiv.org/abs/2311.10122) | [GitHub](https://github.com/PKU-YuanGroup/Video-LLaVA) |
| 2023 | LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models | [Project](https://llama-vid.github.io/) | [arXiv](https://arxiv.org/abs/2311.17043) | [GitHub](https://github.com/dvlab-research/LLaMA-VID) |
| 2023 | TimeChat: A Time-sensitive Multimodal LLM for Long Video Understanding | [Project](https://timechat-open.github.io/) | [arXiv](https://arxiv.org/abs/2312.02051) | [GitHub](https://github.com/RenShuhuai-Andy/TimeChat) |
| 2023 | VideoChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models | [Project](https://mbzuai-oryx.github.io/Video-ChatGPT/) | [arXiv](https://arxiv.org/abs/2306.05424) | [GitHub](https://github.com/mbzuai-oryx/Video-ChatGPT) |
| 2023 | Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding | [Project](https://video-llama.github.io/) | [arXiv](https://arxiv.org/abs/2306.02858) | [GitHub](https://github.com/DAMO-NLP-SG/Video-LLaMA) |
| 2023 | VideoChat: Chat-Centric Video Understanding | [Project](https://github.com/OpenGVLab/Ask-Anything) | [arXiv](https://arxiv.org/abs/2305.06355) | [GitHub](https://github.com/OpenGVLab/Ask-Anything) |
| 2023 | MovieChat: From Dense Token to Sparse Memory for Long Video Understanding | [Project](https://rese1f.github.io/MovieChat/) | [arXiv](https://arxiv.org/abs/2307.16449) | [GitHub](https://github.com/rese1f/MovieChat) |
| 2023 | PandaGPT: One Model to Instruction-Follow Them All | [Project](https://panda-gpt.github.io/) | [arXiv](https://arxiv.org/abs/2305.16355) | [GitHub](https://github.com/yxuansu/PandaGPT) |
| 2023 | NExT-GPT: Any-to-Any Multimodal LLM | [Project](https://next-gpt.github.io/) | [arXiv](https://arxiv.org/abs/2309.05519) | [GitHub](https://github.com/NExT-GPT/NExT-GPT) |

## Grounding, GUI Agents, and Tool Use

| Year | Title | Web | Paper | Code |
|---|---|---|---|---|
| 2025 | MMSearch-R1: Incentivizing LMMs to Search | [Project](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1) | [arXiv](https://arxiv.org/abs/2506.20670) | [GitHub](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1) |
| 2024 | OS-Atlas: A Foundation Action Model for Generalist GUI Agents | [Project](https://osatlas.github.io/) | [arXiv](https://arxiv.org/abs/2410.23218) | [GitHub](https://github.com/OS-Copilot/OS-Atlas) |
| 2024 | OmniParser for Pure Vision Based GUI Agent | [Project](https://microsoft.github.io/OmniParser/) | [arXiv](https://arxiv.org/abs/2408.00203) | [GitHub](https://github.com/microsoft/OmniParser) |
| 2024 | VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks | [Project](https://jykoh.com/vwa) | [arXiv](https://arxiv.org/abs/2401.13649) | [GitHub](https://github.com/web-arena-x/visualwebarena) |
| 2024 | WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models | [Project](https://webvoyager.github.io/) | [arXiv](https://arxiv.org/abs/2401.13919) | [GitHub](https://github.com/MinorJerry/WebVoyager) |
| 2024 | SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents | [Project](https://github.com/njucckevin/SeeClick) | [arXiv](https://arxiv.org/abs/2401.10935) | [GitHub](https://github.com/njucckevin/SeeClick) |
| 2024 | ScreenSpot: GUI Grounding Benchmark | [Project](https://github.com/njucckevin/SeeClick/tree/main/ScreenSpot) | [arXiv](https://arxiv.org/abs/2404.07955) | [GitHub](https://github.com/njucckevin/SeeClick) |
| 2023 | CogAgent: A Visual Language Model for GUI Agents | [Project](https://github.com/THUDM/CogVLM/tree/main/CogAgent) | [arXiv](https://arxiv.org/abs/2312.08914) | [GitHub](https://github.com/THUDM/CogVLM) |
| 2023 | Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding | [Project](https://som-gpt4v.github.io/) | [arXiv](https://arxiv.org/abs/2310.11441) | [GitHub](https://github.com/microsoft/SoM) |
| 2023 | Ferret: Refer and Ground Anything Anywhere at Any Granularity | [Project](https://github.com/apple/ml-ferret) | [arXiv](https://arxiv.org/abs/2310.07704) | [GitHub](https://github.com/apple/ml-ferret) |
| 2023 | LISA: Reasoning Segmentation via Large Language Model | [Project](https://github.com/dvlab-research/LISA) | [arXiv](https://arxiv.org/abs/2308.00692) | [GitHub](https://github.com/dvlab-research/LISA) |
| 2023 | Kosmos-2: Grounding Multimodal Large Language Models to the World | [Project](https://github.com/microsoft/unilm/tree/master/kosmos-2) | [arXiv](https://arxiv.org/abs/2306.14824) | [GitHub](https://github.com/microsoft/unilm) |
| 2023 | Shikra: Referencing and Grounding in Multimodal Dialogue | [Project](https://shikras.github.io/) | [arXiv](https://arxiv.org/abs/2306.15195) | [GitHub](https://github.com/shikras/shikra) |
| 2023 | Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models | [Project](https://github.com/microsoft/TaskMatrix) | [arXiv](https://arxiv.org/abs/2303.04671) | [GitHub](https://github.com/microsoft/TaskMatrix) |

## Domain-Specific Post-Training

| Year | Title | Domain | Web | Paper | Code |
|---|---|---|---|---|---|
| 2025 | On Domain-Specific Post-Training for MLLMs | General domain adaptation | [arXiv](https://arxiv.org/abs/2411.19930) | [arXiv](https://arxiv.org/abs/2411.19930) | - |
| 2024 | Med-Gemini: Towards Generalist Biomedical AI | Medical | [Google](https://research.google/blog/advancing-medical-ai-with-med-gemini/) | [arXiv](https://arxiv.org/abs/2405.03162) | - |
| 2024 | CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation | Medical | [Project](https://stanford-aimi.github.io/chexagent.html) | [arXiv](https://arxiv.org/abs/2401.12208) | [GitHub](https://github.com/Stanford-AIMI/CheXagent) |
| 2024 | OpenVLA: An Open-Source Vision-Language-Action Model | Robotics | [Project](https://openvla.github.io/) | [arXiv](https://arxiv.org/abs/2406.09246) | [GitHub](https://github.com/openvla/openvla) |
| 2024 | Octo: An Open-Source Generalist Robot Policy | Robotics | [Project](https://octo-models.github.io/) | [arXiv](https://arxiv.org/abs/2405.12213) | [GitHub](https://github.com/octo-models/octo) |
| 2024 | LLaVA-UHD: An LMM Perceiving Any Aspect Ratio and High-Resolution Images | High-res vision | [Project](https://llava-vl.github.io/blog/2024-03-10-llava-uhd/) | [arXiv](https://arxiv.org/abs/2403.11703) | [GitHub](https://github.com/thunlp/LLaVA-UHD) |
| 2024 | ChartAssistant: A Universal Chart Multimodal Language Model | Chart | [Project](https://chartllm.github.io/) | [arXiv](https://arxiv.org/abs/2401.02384) | [GitHub](https://github.com/OpenGVLab/ChartAst) |
| 2024 | mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding | Document | [Project](https://github.com/X-PLUG/mPLUG-DocOwl) | [arXiv](https://arxiv.org/abs/2403.12895) | [GitHub](https://github.com/X-PLUG/mPLUG-DocOwl) |
| 2023 | LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine | Medical | [Project](https://github.com/microsoft/LLaVA-Med) | [arXiv](https://arxiv.org/abs/2306.00890) | [GitHub](https://github.com/microsoft/LLaVA-Med) |
| 2023 | Med-Flamingo: A Multimodal Medical Few-Shot Learner | Medical | [Project](https://med-flamingo.github.io/) | [arXiv](https://arxiv.org/abs/2307.15189) | [GitHub](https://github.com/snap-stanford/med-flamingo) |
| 2023 | RadFM: A 3D Radiology Foundation Model | Medical | [Project](https://chaoyi-wu.github.io/RadFM/) | [arXiv](https://arxiv.org/abs/2308.02463) | [GitHub](https://github.com/chaoyi-wu/RadFM) |
| 2023 | PathChat: A Vision-Language Model for Pathology | Medical | [Project](https://www.mahmoodlab.org/pathchat) | [Nature](https://www.nature.com/articles/s41586-024-07618-3) | - |
| 2023 | GeoChat: Grounded Large Vision-Language Model for Remote Sensing | Remote sensing | [Project](https://mbzuai-oryx.github.io/GeoChat/) | [arXiv](https://arxiv.org/abs/2311.15826) | [GitHub](https://github.com/mbzuai-oryx/GeoChat) |
| 2023 | EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought | Embodied AI | [Project](https://embodiedgpt.github.io/) | [arXiv](https://arxiv.org/abs/2305.15021) | [GitHub](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch) |
| 2023 | RoboFlamingo: A Multimodal Generalist Agent for Robotic Manipulation | Robotics | [Project](https://roboflamingo.github.io/) | [arXiv](https://arxiv.org/abs/2311.01378) | [GitHub](https://github.com/RoboFlamingo/RoboFlamingo) |
| 2023 | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | Robotics | [DeepMind](https://www.deepmind.com/blog/rt-2-new-model-translates-vision-and-language-into-action) | [arXiv](https://arxiv.org/abs/2307.15818) | - |

## Evaluation and Benchmarks

| Year | Benchmark | Focus | Web | Paper | Code / Data |
|---|---|---|---|---|---|
| 2025 | Video-MME-v2 | Video understanding | [Project](https://video-mme-v2.netlify.app/) | [arXiv](https://arxiv.org/abs/2604.05015) | [GitHub](https://github.com/MME-Benchmarks/Video-MME-v2) |
| 2025 | BLINK | Visual perception | [Project](https://zeyofu.github.io/blink/) | [ECCV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03356.pdf) | [GitHub](https://github.com/zeyofu/BLINK_Benchmark) |
| 2024 | MMMU-Pro | College-level multimodal reasoning | [Project](https://mmmu-benchmark.github.io/) | [arXiv](https://arxiv.org/abs/2409.02813) | [GitHub](https://github.com/MMMU-Benchmark/MMMU) |
| 2024 | MME-RealWorld | High-resolution real-world scenes | [HF](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld) | [arXiv](https://arxiv.org/abs/2408.13257) | [HF](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld) |
| 2024 | Video-MME | Comprehensive video analysis | [Project](https://video-mme.github.io/) | [arXiv](https://arxiv.org/abs/2405.21075) | [GitHub](https://github.com/BradyFU/Video-MME) |
| 2024 | CharXiv | Chart understanding | [Project](https://charxiv.github.io/) | [arXiv](https://arxiv.org/abs/2406.18521) | [HF](https://huggingface.co/datasets/princeton-nlp/CharXiv) |
| 2024 | Math-Vision | Visual mathematical reasoning | [Project](https://mathvision-cuhk.github.io/) | [arXiv](https://arxiv.org/abs/2402.14804) | [GitHub](https://github.com/mathvision-cuhk/MathVision) |
| 2024 | MathVerse | Visual math reasoning diagnostics | [Project](https://mathverse-cuhk.github.io/) | [arXiv](https://arxiv.org/abs/2403.14624) | [GitHub](https://github.com/ZrrSkywalker/MathVerse) |
| 2024 | TempCompass | Temporal video understanding | [Project](https://llyx97.github.io/TempCompass/) | [arXiv](https://arxiv.org/abs/2403.00476) | [GitHub](https://github.com/llyx97/TempCompass) |
| 2024 | VL-ICL Bench | Multimodal in-context learning | [Project](https://ys-zong.github.io/VL-ICL/) | [arXiv](https://arxiv.org/abs/2403.13164) | [GitHub](https://github.com/ys-zong/VL-ICL) |
| 2024 | MMStar | Multi-modal elite vision-language benchmark | [Project](https://mmstar-benchmark.github.io/) | [arXiv](https://arxiv.org/abs/2403.20330) | [GitHub](https://github.com/MMStar-Benchmark/MMStar) |
| 2024 | RealWorldQA | Real-world image question answering | [HF](https://huggingface.co/datasets/xai-org/RealworldQA) | - | [HF](https://huggingface.co/datasets/xai-org/RealworldQA) |
| 2024 | OCRBench | Text-rich image understanding | [Project](https://github.com/Yuliang-Liu/MultimodalOCR) | [arXiv](https://arxiv.org/abs/2307.02499) | [GitHub](https://github.com/Yuliang-Liu/MultimodalOCR) |
| 2023 | MME | Comprehensive MLLM evaluation | [HF](https://huggingface.co/datasets/lmms-lab/MME) | [arXiv](https://arxiv.org/abs/2306.13394) | [GitHub](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) |
| 2023 | MMBench | Robust multimodal benchmark | [Project](https://opencompass.org.cn/mmbench) | [arXiv](https://arxiv.org/abs/2307.06281) | [GitHub](https://github.com/open-compass/MMBench) |
| 2023 | SEED-Bench | Generative comprehension benchmark | [Project](https://huggingface.co/datasets/AILab-CVC/SEED-Bench) | [arXiv](https://arxiv.org/abs/2307.16125) | [GitHub](https://github.com/AILab-CVC/SEED-Bench) |
| 2023 | MM-Vet | Integrated multimodal capability evaluation | [Project](https://mm-vet.github.io/) | [arXiv](https://arxiv.org/abs/2308.02490) | [GitHub](https://github.com/yuweihao/MM-Vet) |
| 2023 | MMMU | Massive multi-discipline multimodal understanding | [Project](https://mmmu-benchmark.github.io/) | [arXiv](https://arxiv.org/abs/2311.16502) | [GitHub](https://github.com/MMMU-Benchmark/MMMU) |
| 2023 | HallusionBench | Hallucination and illusion diagnostics | [Project](https://github.com/tianyi-lab/HallusionBench) | [arXiv](https://arxiv.org/abs/2310.14566) | [GitHub](https://github.com/tianyi-lab/HallusionBench) |
| 2023 | POPE | Object hallucination | [Project](https://github.com/RUCAIBox/POPE) | [arXiv](https://arxiv.org/abs/2305.10355) | [GitHub](https://github.com/RUCAIBox/POPE) |
| 2023 | LLaVA-Bench | Open-ended visual chat | [Project](https://llava-vl.github.io/) | [arXiv](https://arxiv.org/abs/2304.08485) | [GitHub](https://github.com/haotian-liu/LLaVA) |
| 2023 | MVBench | Multi-modal video benchmark | [Project](https://github.com/OpenGVLab/Ask-Anything) | [arXiv](https://arxiv.org/abs/2311.17005) | [GitHub](https://github.com/OpenGVLab/Ask-Anything) |

## Related Awesome Lists

| Name | Scope | Link |
|---|---|---|
| Awesome-Multimodal-Large-Language-Models | General MLLM papers, datasets, and benchmarks | [GitHub](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) |
| MM-LLMs | Recent advances in multimodal LLMs | [Project](https://mm-llms.github.io/) |
| Awesome-LLM-Post-training | General LLM post-training and reasoning | [GitHub](https://github.com/mbzuai-oryx/Awesome-LLM-Post-training) |
| Awesome-Large-Multimodal-Reasoning-Models | Multimodal reasoning models and RL-style methods | [GitHub](https://github.com/HITsz-TMG/Awesome-Large-Multimodal-Reasoning-Models) |
| Multimodal R1 Papers | GRPO/RL-inspired visual and multimodal reasoning | [GitHub](https://github.com/gullalc/multimodal_r1_papers) |

## Maintenance Notes

- Prefer official project pages and repositories.
- For papers with no released code, keep the code cell as `-` instead of linking unofficial reimplementations.
- For datasets, use the official repository or Hugging Face dataset card.
- Add new entries under the method section that best matches the post-training signal: SFT data, preference data, reward model, RL/RLVR, hallucination mitigation, reasoning, tool use, or evaluation.

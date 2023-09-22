# Autoformalization Benchmark Creation Using Human Preferences and a Reward Model

## Supervisor:
Brando Miranda, contact email: brando9@stanford.edu
## Field:
AI/ML
## Academic Year:
Aut_win_spr, 2023-2024
## Course Credit:
Available
## Team Size:
Up to 5 students
## Project Description:
Autoformalization is pivotal for converting informal statements, typically in natural language, into formal, verifiable statements, such as those in Python, Lean, Coq, or Isabelle. This project aims to establish a benchmark for autoformalization derived from Lean's Mathlib library, incorporating human judgments to assess the quality of formalization or informalization. Initially, data will be manually created, resembling the structure of the debug1_af data, with formalization executed using advanced models like GPT-4 or Claude, followed by expert human evaluation on the quality of formalization and informalization. The goal is to label at least 500 examples to train a reward model, drawing insights from the LIMA paper on optimizing model performance with limited examples. Subsequently, the trained reward model will be evaluated for its alignment with human preferences and utilized to label the entire paired data set, providing two scores for evaluation and training. This endeavor is motivated by the aspiration to construct an automated mathematician capable of unlocking the vast reservoir of knowledge embedded in mathematical textbooks written in natural language, thereby contributing to advancements in mathematics, scientific discovery, and AI safety through autoformalization.
## Recommended Background:
Interested candidates are requested to share their background when reaching out.
## Prerequisites / Preparation:
Contributors to this project should be proficient in Python to make direct contributions to the project.
## Key Citations:
1. Autoformalization: https://arxiv.org/abs/2205.12615
2. LIMA - Less is More For Alignment: https://arxiv.org/abs/2305.11206
3. ProofNet: https://arxiv.org/abs/2302.12433
4. ProofNet Dataset: https://huggingface.co/datasets/hoskinson-center/proofnet
5. LeanDojo: https://github.com/lean-dojo
## Motivation:
The overarching vision of this project is to pave the way for the realization of an automated mathematician, capable of automating mathematics, scientific discovery, and ensuring AI safety through formal mathematics. The conjecture is that formal mathematics is indispensable for creating safe AGI, as safety necessitates a universal quantifier asserting the impossibility of AGI causing harm to humanity. This project is a step towards harnessing the untapped potential of information contained in mathematical textbooks written in informal language through the process of autoformalization.

# Experiments

## Experiment 1: Build paired MathLib data using some AF model e.g., GPT4, Claude (Api or we build our API to the webversion) or HF model

## Experiment 2: Manually Score some Formal Informal Pairs about how good the informalization/formalization are

## Experiment 3: Can LIMA inspire us how to build our human preferences data/socring for how good a formalization/informalization is?

## Experiment 4: Build a Reward model based on our data & evaluate it
- train it on the manually scored data
- have it generate scores and self train it on it
- use other paired data (e.g., lean docs, py docs) and train it with that
- strucutred NL -> FL, lean, Chris Hahn

- evaluate it:
- Are the scores of our reward model close to the ground truth human annotations? 

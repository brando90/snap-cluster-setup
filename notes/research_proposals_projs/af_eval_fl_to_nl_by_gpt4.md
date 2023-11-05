# Evaluating AutoFormalization using high quality LLM as a reference

Idea: Create a eval benchmark for Autoformalization (AF) with pairs as follow

- `FL, NL = GPT4(FL)`

where `FL \in ITP` library (e.g., mathlib, proverbot9001, Isabelle?) and NL generated is from a very high quality LLM ( e.g., GPT4 or combination of them)

Motivation: 
- only for eval (no training since GPT4 is priorpiratery)
- large scale, easy to make and gives at least PPL/CE loss feeling if it's as good at least as GPT4
- GPT4 seemed good at informalization so this task is high quality in the NL statement.

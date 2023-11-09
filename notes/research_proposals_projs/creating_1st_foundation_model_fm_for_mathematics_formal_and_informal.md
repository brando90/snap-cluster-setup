# Creating first FM for Mathematics (informal & formal)
Idea: train with all mathematical knowledge, both formal and informal. 
But mostly informal first and then use a model that extracts all the formalizable informal statements.
This would be defenitions, theorems, lemmas, proofs, etc. and convert them to formal mathematcs (lean, coq, isabelle).
Then we create the best model that knows all mathematics.
We also train it with all the proof data we can e.g., proof terms, holes, tatic proofs, all type of aligments e.g., proof state <->informalizaiton.
Pruning for high quality data is valuable e.g., with my data aligment/quality/diversity metrics, the kernel, type checker, everything possible, reward models too.



- Goal: 
	- create system/infrastructure (e.g., using langchains, etc) to translate all informal mathematical knowledge to formal mathematical knowledge
	- create the data set needed to create the first foundation model for formal mathematics
		- (I am aware I can't train one. If I do create this data set then I can figure out who to collaborate with. I hope to at least get an impressive fine-tuned model. Not sure which FM to train but leaning to falcon family. Open Minerva?)
	- (create a playground to prove something truly impressive)
- Problem Statement:
	- There is not enough formal mathematical data to train a excellent model like GPT-4
	- some FL
	- lots NL -> FL
	- unpair
- Hypothesis/Assumption:
	- Machine Learning models are mainly good an interpolating not extrapolating because ERM is built for interpolation. 
	- Therefore I hypothesize we need some neuro-symbolic (e.g., search) symbiosis eventually e.g., sketching, search/MCTS,
	- However, I conjecture without a powerful foundation-like starting points, most symbolic/search/sketch augmentations won't succeed because they could be made irrelevant by a powerful learning model/method. 
	- bootstrap from tiny extrapolation length (TODO)
- Impact: 
	- Due to my conjectured limitation of ML as a field itself in the context of fields that require extrapolation -- in this case formal reasoning -- then addressing that limitation head on is non-trivial fundamental contribution to the field. 
- Method: 
	- Goal: Generate a massive formal (and informal) mathematics data set from translation from informal mathematics for pre-training a MathFM. 
		- data points = 
			- dpt = <IM/NL, FM/FL (THM), Pf>
			- dpt = <NL, NLPf, FL (THM), [Pf]_i>
	- Summary Method: translate informal mathematics to formal automatically with minimal (human intervention) such that proving accuracy increases with the aid of pruners & self-improving AF
		- data crawlers
			- DC1: get mathematical knowledge textbooks, math SO, arxiv informal/latex, using LLMs (langchain?)
			- DC2: NL -> Latex (Coq sys) -> Coq
			- DC3: data already available for AF
			- (DC1': reward model indicating good autoformalizations/translations)
		- pruners for generated <NL, FL> from NL -> FL:
			- P1: ITP -- the ITP (Coq/Lean/Isabelle). Only include Autoformalizations that pass syntax & type checks, proved by hammers/other provers
			- P2: PE -- PosEval (proof length, provability, truth, equivalent to false -> true)
			- P3: DB -- increases the prove accuracy of an external proof data base (that I'm creating but for concrete example CoqGym, lean MathLib, MiniF2F, Tony's AF ds task NL -> FL)
			- P4: DQ -- automatic/t2v DQ filter i.e., get autoformalizations that align with val/test set.
				- (no mostly likely due to P4, P5': reward model that takes in data set or embedding of dataset and predicts if it improves proof accuracy)
		- self-improving loop
			- .... TODO
- Evaluation:
	- Eval1: our DB (extended CoqGym/UPyCoq)
	- Eval2: Flag posting with impressive theorems e.g. FLT, PvsNP
	- Eval3: past DS (MiniF2F)
	- (Eval1': proof length/compression)
- Limitation:
	- figure out how you'd use this augmented FMaths data set. For now mainly through; 1. in context learning (ICL) 2. fine-tuning (ft)
	- if DB is saturated (that's good! if it's non-trivial e.g. coq-gym), then we can't know if we are making progress in improving. 
		- what could we address this limitation? 
		- 1. use MAF-sys (Massive AutoFormalizer system) to autoformalize a collection of impressive/non-trivial human curated statements for a new DB e.g. FLT, P vs NP + something impressive that doesn't give sparse rewards/signal
		- 2. increase DB by including all evals for FMaths
		- 3. Think wider than just Coq/Lean (ITPS)?

- Tony's data set AF https://docs.google.com/spreadsheets/d/1wP4rRm-yboh7GVbh6sl8aoz2RqF7rybyidATbQdxBs8/edit#gid=1063821127 
	- selected stmt
		- connected 15/73
		- convex 4/148
		- polynomial_factorial 11/47
	- this data set seems for expensive to get. My feel is that it's best use is to use it as an eval set/db for pruner 3 . 
	- 

---
take aways/conclusions

- Tony is worried that the autoformalization his system does are not useful for proving
	- didn't fully understand if he was worried about the translation being false or something else
	- I can probably reverse engineer what the issue is if I think more carefully what the round trip does
	- loop 1: NL -> FL (training I assume he means)
		- helps model understand mathematics
		- i.e., this skips the RL inefficiency since we train the model with the ultimate state, actions pairs what decision to take (for a truly difficult env that I don't think any time soon anything like alphazero is possible)
	- loop 2: FL -> NL (training I assume)
		- **this helps get more autoformalization data, given that is ALSO limited**
		- helps loop1 have a higher chance of doing translation that are
			- 1. correct since they come from real FL
			- 2. 
		- well this gives us more NL pairs (even multiple) for FL that are genuinely new & the pairing is likely high quality. So perhaps it helps loop1 to generate NL -> FL better, since we don't actually have enough <NL, FL> in practice even in the original FL \in RealHumanWritten Code e.g. there aren't enough comments
		- (this can help us create a data set for just AF)
		- method:
			- generate <FL, NL> using good FM
			- these are new pairs. We can improve the Autoformalizer (NL -> FL) given that we now have new pairs
	- Q: can we validate Tony's worry? What if loop 1 already works to improve proving accuracy? Should we really be worried.
	- Q: I think we can train a reward model with this
- I don't "understand" why the round trip is expensive or what the issue is with it
	- my guess is that the pre-training/fine-tuning for both directions is expensive? 
	- I think just a concrete evidence (even if just with a few back of the envelope numbers), would help me understand
- TODO of details
	- what target theories to do
	- even just real analysis and topology would be amazing
		- perhaps let's do analysis, Rudin text book? (sort of familiar and have hand annotated examples)

- Qs
	- is using GPT4 for translation a good idea?
		- pros: probably really good data set, can validate/falsify idea
		- cons: expensive, cannot be used for commerical use

---

- other ideas
	- write poseval -> proof_length + truth/false + provable/unprovable/unk?
	- improve autoform via latex e.g, NL -> Latex (coq sys) -> CoqLang
	- reward model, hand annotations are only good for eval to asses/eval the translation/llms performance in an imperfect task like AF not AF4ProofAcc
	- Reverse autoformalization <FL, NL> . Then use that data get data point for autoformalization (?)
		- loop1: NL -> FL
			- 
		- loop2: FL -> NL
			- <NL, FL> 
			- 
		- loop 3: FL -> NL -> FL
			- 
		- -> increase likelihood not predicting false
		- unpaired matching problem
			- img <-> text synthesis
		- template to informal language
	- prover prompt itself
	- Models (might be same LLM)
		- HR model
		- proof accuracy model
		- prover model
		- autoformalization mode
	- tasks
		- T1: Autoformalization, <NL, FL> from humans
		- T2: proof accuracy reward <NL, FL> -> R
		- T2: More proof data <FL, Pf>
	-  make a model that is useful to start creating a useful data set for autoformalization **
		- make it good enough at autoformalization such that people want to use it and prompt it
	- collabs with mathematicians (?) at SU?
	- hard to beat the GPT-4 teacher model
	- Train an LLM to NL->FL via a Differentiable Reward function
		- sort of like RLHF
	- ExpIt to improve sampling more options of the AF
	- if new FM aligns with val set, then add (instead of task2vec + proof acc model that predicts if it will improve proof acc)
	- human learned reward model for a good translation of <NL, FL>
	- Pos Eval model (proof term length, tactic length e.g. chars) 
		- plus provability, false, unkn
		- M7: from multiple translation of <NL, FL> that type check, select one the ones that have shortest proof length/most proof acc
	- (from Chris's talk I don't think de bruijn will help as much as I wished...? due to the arbitrary char used it just learns to re-use the one from the context (likely), so removing it and just putting the idx like in de bruijn might not help as much?)
	- AF free service for proj/crowd source/SU maths teamup
	- ----
	- template of NL <-> FM (synthetic transations)
		- forall == "for all, for every, for any"
		- exists == "for some, there exists"
		- - theory specific? per coq proj?
		- thm <-> nl
		- more auto template, Lina's Company?
	- Problem Statement: no MathFM
		- P1 not enough <nl, fl> pairs, 
			- hard to construct & for real correctness we need human intervention
		- P2 also not enough <thm, pf>, 
			- the thm has to be correctly constructed by human + generating proofs is hard
	- ExpIt
	- Provers
		- fm Prover
		- Proverbot
		- tactician
	- + other combinations
	- ----
	- truly indepedent fully auto maths that improve itself (chris H)

---

- rough notes  [x] 
- go through past notes on method  [x] 
	- write poseval -> proof_length + truth/false + provable/unprovable/unk?
- write motivation well (radical honesty)
- write methods section well
- eff

- plan

---


- ess
	- Question: is this project worth pursuing even if the only thing that comes out is a 
- effortless


- Qs
	- data already available
	- number of tokens trainable in academia (crfm)
		- pt
		- ft
	- what FM/LLM to use
		- math-lm
		- falcon family
	- problem selection
		- autoformalization for the sake of it
		- autoformalization for proving something impressive (flt, p vs np)
	- autoformalization benchmark for the sake of autoformalization
	- should I use GPT4?
		- then it can't be used for commercial use by anyone...does this really move the field forwrad
		- but for research then it does create a cool data set
---


- plan
	- FL -> NL -> FL
		- no env
	- AF
	- AF_DB 
		- ppl, str match, avg str match
		- acc increases
	- --
	- Test: MAF
		- train = textbooks
		- eval = textbook
	- AF(textbook) -> <NL, IPf, FL, FPf>
		- NL

---

- plan
	- train AF/MathLLM: 
		- pre-train:
			- FL -> NL
		- idea: FL -> NL -> FL
			- c idea1: [AF, FL] -> [IF, NL] -> FL
			- c idea2: [AF, FL] -> [IF, sNL] -> FL
		- AF train data set
			- coqgym proverbot9001
			- PISA
			- (LeanDojo)
		- HF trainer + train loop
	- Eval: AF_DB 
		- idea 1: ppl, avg str match, token edit distance, str match,
			- NL -> FL
			- FL -> NL
		- idea 2: tony gave us ground 
			- prev metrics, eval on it
			- https://docs.google.com/spreadsheets/d/1wP4rRm-yboh7GVbh6sl8aoz2RqF7rybyidATbQdxBs8/edit#gid=1063821127
		- idea 3: 
			- TODO: gpt4, human curated, acc increases
	- --
	- Test: MAF
		- train = textbooks
		- eval = textbook
	- AF(textbook) -> <NL, IPf, FL, FPf>
		- NL
 
https://www.evernote.com/shard/s410/sh/17977b8b-882c-a2ae-cd77-bbc51e8c2333/SAXGZxAjXZWMzSnFFw71YjJwK-Q1_koF7lcDWT7Nrw1EVXPuiHwQzovo5A


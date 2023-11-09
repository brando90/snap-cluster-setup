# Massive AutoFormalization (MAF)
massive autoformalization for the first automated mathematician

## Prompt use to Autoformalize (AF) to Isabbele by Yuhuai Wu et al. 

```python
prompt_nl_2_fl_isa : str = f"""Natural language version: {nl_stmt} \n
Translate the natural language version to an Isabelle version:
"""
```

[//]: # (https://chat.openai.com/share/7dbd76bf-c42f-4fcc-a0ec-09eb452380dd)
## Dual-AF-PreTrain - Dual Informal-Formal Back Translation Training for AutoFormalization (Train Algorithm 1)
other name option: Dual Grounded Back-Translation for Massive AutoFormalization (DGBT4MAF)
other name option: Jointly Grounded Back-Translation for Massive AutoFormalization (JGBT4MAF)

Goal: Build an agent that can AutoFormalize a textbook.

Problem: Lack of pairing data for translation for informal mathematics IM (also notated as NL for natural language) and formal mathematics (also notated as FL for formal language). 
I will use IM==NL and FM==FL as equivalently 

For that we hypothesize one needs to:
1. Have an AF model that can translate informal to formal mathematics - the hard direction e.g., due to humans writing ambiguous natural language, missing information that are infered by humans mathematical common sense, etc). Noted as: `nl->fl`
2. Have an AF model that is used to the type of informal mathematics written by humans e.g., it can translate from real informal mathematics and not only from some synthetically created (nl, fl) pairs

Idea: train on both, i. `fl* -> [nl_i]_i -> fl*` and ii. `nl* -> [fl_i]_i -> nl*`. The `*` denotes high quality target sequences e.g., informal text written by mathematicians (nl*), formal text from an ITP library like mathlib (nl*). 
- 1. Learn to Formalize: Via the second step, `[nl_i]_i -> fl*`, the model learns to map ambiguous natural language to formal language. Crucially, the ground truth formal language is used from a formal language dataset e.g., mathlib. This step can be done first to make sure the model is grounded in outputting correct translation.
- 2. Learn to Informalize: Via the first step `nl* -> [fl_i]_i` the model learns to autoformalize the type of text it will encounter mathematical texts written by human mathematicians. 

Note: doing all steps together might be fine too for better usage of GPU.
Note: `[nl_i]_i`, `[fl_i]` is obtained via sampling. Current suggestion is nucleus/top-p sampling (out of the box in HF).

```python
def train_to_af_for_maf(mdl : causal_lm,
                        formal_data_set, # e.g., ITP lib like mathlib
                        informal_data_set,  # e.g., time-tested maths textbook e.g., Rudin, CLRS.
                        ):
    for (nl, fl*) in formal_data_set; for (nl*, fl) in informal_data_set;
        # -- Learn to Formalize: nl_i->fl* from fl* -> [nl_i]_i -> fl*
        [nl_i]_i := mdl("informalize " + fl*, sampling=top_p, num_out=k)  # noise is good for robustness!
        [fl_i]_i := mdl("formalize " + [nl_i]_i, sampling=random, num_out=1)  # num_out = 1 => learns equivalences! todo: num_out=1 good?
        # - Train Step to Formalize from high quality formal dataset ~ opt.step((nl_i -> fl*)_i) 
        loss = loss_fn([fl*, fl_i]_i); loss.backward() 
        loss = lss_fn([nl, nl_i]_i); loss.backward() if nl != '' else None
    
        # -- Learn to Informalize: fl_j->nl* from nl* -> [fl_j]_j -> nl*
        [fl_j]_j := mdl('formalize ' + nl*, sampling=top_p, num_out=k)
        [nl_j]_j := mdl('informalize ' + [fl_j]_j, sampling=random, num_out=1)
        # - Train Step to Informalize from high quality informal dataset ~ opt.step([fl_j -> nl*])
        loss = loss_fn([nl_j, nl*]); loss.backward()
        loss = lss_fn([fl, fl_j]_j); loss.backward() if fl != '' else None
    
        # -- Predict Proofs given Theorem (proofs are hard to come up with, so whenever available learn to generate them)
        # - IT -> IPf (half round trip training) todo: careful with predicting only sledgehammer
        ipf_i = mdl('generate informal proof ', it)  # model can infer it's from an informal theorem given the theorem will be in nl so I think this shorter prompt is better, we are fine-tuning anyway.
        loss = loss_fn([ipf_i]_i, ipf*); loss.backward()
        it_i = mdl('generate informal theorem ', it)
        loss = loss_fn([it_i]_i, it*); loss.backward()  # just autoregressive loss, so it knows the theorem. I've learned that ICL isn't actually as good as fine-tuning based on my daily use of GPT4/Claude 2.0. 
    
        # - FT -> FPf (half round trip training) todo: careful with predicting only sledgehammer
        fpf_i = mdl('generate formal proof ', ft)  # model can infer it's from an form theorem given the theorem will be in fl, shorter prompt better likely + we are already fine-tuning. 
        loss = loss_fn([fpf_i]_i, fpf*); loss.backward()
        ft_i = mdl('generate formal theorem ', ft)
        loss = loss_fn([ft_i]_i, ft*); loss.backward()  # just autoregressive loss, so it knows the theorem. I've learned that ICL isn't actually as good as fine-tuning based on my daily use of GPT4/Claude 2.0. 
    
        # -- Standard Autoregressive training: Predict raw formal and informal texbook data (pre-pre-training)
        loss = loss_fn(mdl(fl*), fl*); loss.backward()
        loss = loss_fn(mdl(nl*), nl*); loss.backward()
        # loss = loss_fn(mdl(fl), fl); loss.backward()
        # loss = loss_fn(mdl(nl), nl); loss.backward()

        # -- Jointly train everything (for better hardware usgae)
        opt.step() # trains all tasks: nl->fl, fl->nl, it->ipf, ipf->it, ft->fpf, fpf->ft 
        opt.zero_grad()  # zero grads of only params the opt is optimizing
    
        # -- Stop when target its/num_tokens met
        stop(it == target_its)
    return mdl # for clarify of code, but it likely mutates it :( oh no side-effect...

if __name__ == '__main__':
    # Train with AF4MAF Back Translation based
    mdl = train_to_af_for_maf(mdl, formal_data_set, informal_data_set)

    # Eval if model improved from train procedure for MAF
    print('---- Display if our AF4MAF training improved eval metrics on benchmarks ----')
    eval_af(mdl, eval_data_set=af_dataset, metrics=(ppl, exact_str_match, avg_acc, token_edit_distance))
    eval_proof_acc(mdl, prover=dsp_magnus_hammer, formal_dataset_val, metrics=(proof_accuracy, ppl, exact_str_match, avg_acc, token_edit_distance))  # soft metrics (not proof acc) to see if we get a weaker signal from other metrics on target fl*
    eval_maf_proof_acc(mdl, prover=dsp_magnus_hammer, textbook=sipser_complexity_theory, metrics=(proof_accuracy, ppl, exact_str_match, avg_acc, token_edit_distance))  # proof accuracy checks if the (autoformalized) formal theorem (the human/gpt4 checked) is proved. But the soft/nlp metrics  are used on the informal proof from the textbook to see if model improves the prediction on at least that
```
note: sampling methods for formalize vs informalize might need to be different.
note: additional training data, algin doc strings in normal code (python) to get more <NL, FL> pairs via DGBT-MAF.
Improvements in quality (alignment with target task) can be done by converting functions/defs to have for all stmts e.g.,
<lambda: x : body, doc> -> <forall x, body(x), doc>. Or model trained to know this simple "translation" rule from code to ITP-like lang.
Additional set conditions can be written to exlcude some x's based on reading body e.g., forall x: Int x != 0 Body(x).
Proof terms could help from Lean.

Question: Which is harder formalize or informalize?
Answer: Hypothesis is formalize! todo, write thoughts...

Question: Should we train to formalize on the eval textbook used for MAF eval?
Answer: Brando suggests either yes or no is good. 
For "yes", then only the theorem statements, as long as we don't train on the autoformalized proofs.
For "no", then it's a stronger test our autoformalization tool generalizes.
Likely "no" is the strongest answer, unless we can't get our model to prove the entire textbook. 
Future work can be to remove this requirement.

Question: How to deal with definitions?
Answer: ...first articulate problem well...

idea: more sketch data proof terms <-> isar translations.

## AutoFormaliztation Training Method 2: End-to-End training
TODO: write better & get reasonable starting point/details right. 
```python
fl* -> snl -> fl* (end to end diff, e.g. gumballsf, or just soft tokens output of sf)
nl* -> sfl -> nl* (end to end diff, e.g. gumballsf, or just soft tokens output of sf)
```

## Eval: Massive AutoFormalization (MAF) 

Goal: proof of concept that we could autoformalize all mathematical knowledge. 
Concretely, we propose to autoformalize and prove mathematical textbook of choice 
e.g., Rudin, Sipser's Complexity book, Jay Cumming's Proofs book with COT, (CLRS), etc. 

There are two ways to evaluate if the model is AutoFormalizing and proving the textbook:
1. Eval via informalization-to-formalization (half-round trip) (Task 1/T1): `FT <- IT* -> IPf -> FPf => ITP(FT, FPf) && equivalent(FT*, FT)`
2. Eval with by formalizing theorem and using a prover to prove it (Task 2/T2): `IT -> FT => prover(FT, ITP) && equivalent(FT*, FT)`
where `equivalent(FT*, FT)` tries to prove the equivalence between the sparsely curated textbook theorems `FT*` vs the model generated
theorems. 

Ideal (eval) API:
```python
from pdb import set_trace as st

def autoformalize_textbook_with_maf(textbook: Doc, 
                                    model: PreTrainedModel,  #actual HF mdl type
                                    itp_env: Union[PISA, LeanDojo, UPyCoq],
                                    prover: Union[DSP_MH, MH],
                                    ) -> Pandas_MAF_DataFrame, Raw_Text_DataFame:
    """ """
    # -- columns: thm pf sketch def lemma example exercise solution remarks img img_as_text page_num original_doc_source fthm pff ... verified importance_weight (text before) (text after) & all formal version of it e.g. fthm fpf fsketch fdef etc., decided all informal and formal data same place to avoid issues btw matching/coordinating two different data frames # todo: put this in a proper place to document data bases
    pandas_maf_df, raw_text_df = autoformalize_fragment_with_maf(textbook, model, itp_env, prover, pages=[0-2]) # loop through all textbook when working
    return list_thms

def autoformalize_fragment_with_maf(textbook: Doc,
                                    model: PretTrainedModel,
                                    itp_env: Union[PISA, LeanDojo, UPyCoq],
                                    prover: Union[DSP_MH, MH],
                                    pages: Optionap[Union[list[int], Page_Regex]] = 'maf_all',
                                 # chapters: Optional[Union[list[int], Chapter_Regex]] = "maf_all",  # likely simpler to only support pages?
                                 ) -> Pandas_MAF_DataFrame, Raw_Text_DataFame:
    """"""
    pandas_maf_df, raw_text_df = df.append(new_row_maf, ignore_index=True), df.append(new_row_text, ignore_index=True)
    for page in pages:
        # -- Get Text from Textbook
        raw_textbook += pdf_2_text(textbook, model) if textbook.is_pdf else textbook
        # -- Get & populate Informal & Formal data from textbook text
        pandas_maf_df += populate_pandas_maf_dataset(page, raw_textbook, model, itp_env, prover)  # populates thm pf sketch def ... fthm fpf fsketch fdef etc...
        # -- Get & populate informal only data set too from text in textbook
        raw_text_df += populate_raw_text_dataset(raw_textbook)  # populates all text, especially surrounding text  todo: think how to interleave this with MAF data for better training i.e., model knows context in addition to formalization
    return pandas_maf_df, raw_text_df

def populate_pandas_maf_dataset(page: int,
                                raw_textbook: Raw_Text_DataFame,
                                model: PreTrainedModel,
                                extractor, 
                                itp_env: Union[PISA, LeanDojo, UPyCoq],
                                prover: Union[DSP_MH, MH],
                                ) -> Pandas_MAF_DataFrame:
    """ """
    # -- Get Thm
    autoformalize_fragment(raw_textbook[page], model, extractor, itp_env, prover)
    # -- Get Pf
    # -- Get Sketch (or generate it)  # we should annotate if it's generated by model or by human (e.g., sipser has sketches)
    # -- Get Defs

def autoformalize_fragment(page_as_text, model, extractor, itp_env, prover): 
    """ """
    
    # -- Extract Informal Theorems, Proofs, Sketches, Defs, etc. from fragment of textbook
    # thms, pfs, sks, defs, [not essential: lemmas, egs, exs, solns rks, imgs, imgs_text] = extractor(text, model)  
    thms, pfs, sks, defis, _, _, _, _, _, _, _ = extractor(page_as_text, model)  
    st()  # verify manually this looks good with print
    
    # -- Formalize Informal Theorem, Proof, Sketch, Defs, etc. from fragment of textbook  # todo: note even if the formalization doesn't "parse", might it be benefifial for the model learning with our algorithm?
    # thms, fthms, pfs, fpfs, sks, fpsks, defs, fdef, _, _, _, _, _, _, _ = extractor(page_as_text, model) 
    for thm, pf, sk, defi, in zip(thms, pfs, sks, defs):
        # - Autoformalize
        fthm, fpf, fsk, fdef = formalize(thm, pf, sk, defi)
        print(fthm, fpf, fsk, fdef)
        informal_assert informal_equivalence_check(fthm, thm, itp_env)  # model convinces itself translation is correct or even with a teacher model
        st() # verify manually this looks good with print, e.g. copy paste to my ChatGPT app manually, chat with ChatGPT to check it
        # - Check thm: syntax check, could try equivalence proof if this has already been formalized before e.g., Rudin Real Analysis vs Isabelle Real Analysis eq thm, manually get that thm?
        # assert check(fthm, itp_env)
        print(thm, thm)
        st()
        # - Check pf: likely most important check. If this test passes it most likely will improve proof accuracy of external data base/reward model. note: not worried about False -> True since model is auformalizing from human proofs, not learning to potentially exploit a the itp env with RL
        # assert type_check_formal_proof(fthm, fpf, model, itp_env, prover)
        # assert prove(fthm, model=None, itp_env=itp_env, prover=sledgehammer)
        # assert prove(fthm, model, itp_env, prover)  # DSP_MH
        # - Formal Sketch ... (partial proof term in Coq/Lean, Isar in Isabelle)
    return thms, fthms, pfs, fpfs, sks, fpsks, defs, fdef, _, _, _, _, _, _, _  # todo: should this take
```
note: might also be a good idea to train on the raw text from the textbook while we are at it. Likely would help the model
understand the context and mathematics better, especially for conversing with the model 
e.g., maybe for debugging the MAF process, later on people will really care to chat with the model to **understand** the
formally verified mathematics.

```markdown
- Testing: MAF
    - pre-train
      - on ITP formal language
    - pre-train
        - use Alg. 1 (DGBT4MAF)
        - check if  acc/ppl/token-edit on good AF db improves (e.g., manually curated + gpt5 data used as val signal)
    - collect train from -> textbooks
        - could be same as eval but with train flag, so we can weight differently (formally or informally) verified data.
        - different options e.g.,
            - 1. use it in an online fashion to boostrap and grow data base, or
            - 2. only grow data base every so often and (re) train with DGBT4MAF
    - eval = textbook
        - same as train but with eval flag on
- AF(textbook) => <NL, IPf, FL, FPf>
```

idea: use an LLM to extract the theorems and proofs. If the textbook if semi-structured (e.g., Thm: ... Proof. ... [] or Qed) then the extraction of the informal Thm, Pf pair might be easier!
Relevant work:
- Simran Arora's Evaporate: https://github.com/HazyResearch/evaporate (they explicitly quote how their method is cheaper $$ than GPT-4 API)
    - note: we only need extraction, not (document/textbook) retrieval most likely.
- Was advised against LangChains (Rylan Shaeffer): https://twitter.com/AravSrinivas/status/1677884199183994881?t=K7ngp0m3GmRgnj0RjqThZw&s=19 (<- former Open AI guy, now making a start-up, talking about why not to use LangChain) + More on HN https://news.ycombinator.com/item?id=36645575.
More on HN https://news.ycombinator.com/item?id=36645575
- Q: Should we build API compatible with OpenAI API?

Note:
- Retrieval is like I ask you a question and you retrieve documents/information from the internet/textbooks, etc to answer it []
- Extraction is like I look at this arxiv paper (a specific document) and want to extract the title from it or something

### other useful stuff

- prompts for writing papers: https://github.com/brando90/ultimate-utils/tree/master/prompts/writing_ml_research
- overleaf link: https://www.overleaf.com/9541275884kggkytvcgqgv

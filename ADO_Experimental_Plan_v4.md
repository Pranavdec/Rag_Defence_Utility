# **ADO Sentinel-Strategist: Complete Experimental Specification**

**Target Venues:** USENIX Security, IEEE S\&P, ACM CCS, NeurIPS, ICLR

## **Executive Summary**

This document is the ground-up experimental specification for the paper on Adaptive Defense Orchestration (ADO) for RAG security. **Strategic Framing:** This evaluation is designed to empirically prove the existence of the *security-utility paradox* in realistic RAG deployments, and to demonstrate that dynamic orchestration (ADO) offers a viable, directional step forward. ADO is evaluated as a proof-of-concept for bridging this gap, not as a flawless or production-ready panacea against all future adaptive attacks.

## **Part 1: Dataset Suite and Realistic Corpus Scale**

### **1.1 Corpus Construction Requirements**

To properly demonstrate the security-utility paradox, the retrieval environment must reflect real-world noise.

**The Ingestion Pipeline:** \* Chunk size: 512 tokens

* Overlap: 50 tokens  
* Embeddings: all-MiniLM-L6-v2  
* Vector DB: ChromaDB  
* Top-k retrieval: k=5

**Distractor Injection (CRITICAL):**

For every evaluated dataset, the vector database must contain **both** the gold evidence passages **and a large-scale distractor corpus** (target scale: 50,000–100,000 chunks) to simulate realistic retrieval crowding. Distractors must be domain-matched.

### **1.2 Primary Datasets**

| **Dataset** | **Domain** | **Role in Paper** | **Attack Coverage** | **Distractor Source** |

| **Natural Questions (NQ)** | Open-domain QA | Utility baseline \+ Poisoning | Poisoning | MS MARCO passages (v2.1) |

| **TriviaQA (RC split)** | Trivia / QA | Primary multi-vector setting | Poisoning \+ MIA \+ Leakage | Wikipedia 2021-01-01 snapshot |

| **PubMedQA (pqa\_labeled)** | Biomedical | Domain-shift \+ Privacy | MIA \+ Leakage | PubMed abstracts |

| **FinanceBench** | Enterprise | High-stakes domain | Poisoning \+ Leakage | SEC EDGAR 10-K/10-Q filings |

### **1.3 Query Splits (Pre-generated with fixed seeds)**

* **Benign utility queries:** 200 queries (measure utility under defense policies).  
* **Poisoning trigger queries:** 100 queries (measure Attack Success Rate).  
* **MIA probes:** 200 member / 200 non-member probes (400 total per dataset to ensure strong Clopper-Pearson bounds).  
* **Leakage attack prompts:** 100 prompts (measure verbatim reconstruction).  
* **Trust-farming warm-up queries:** Variable pool of benign queries to farm  S\_trust..

## **Part 2: Attack Suite and Adaptive Adversaries**

The attack suite relies strictly on cited, literature-backed attack families, plus one adaptive adversary to stress-test the stateful nature of the defense.

### **2.1 Static & Baseline Attacks (Blind Adversaries)**

* **Data Poisoning:** PoisonedRAG. Inject adversarially crafted documents at defined rates: 0.01%, 0.05%, 0.1%, and 0.5% of the overall corpus size (e.g., approx. 5, 25, 50, and 250 chunks in a 50k corpus) per trigger query to establish the detection threshold. (Evaluated on NQ, TriviaQA, FinanceBench).  
* **Membership Inference (MIA):** Mask-Based Attack (MBA) masking 5 tokens per probe, and RAG-MIA. Proxy likelihood estimation via GPT-2-XL. (Evaluated on TriviaQA, PubMedQA).  
* **Content Leakage:** Spill-the-Beans and IKEA. Anchor queries steer retrieval, command queries induce extraction. (Evaluated on TriviaQA, FinanceBench, PubMedQA).

### **2.2 The Adaptive "Trust-Farming" Adversary**

To evaluate the Global Trust Score (S\_trust) honestly, we introduce an adaptive adversary that *knows the defense exists and operates a trust score*.

* **The "Fooling" Protocol:** The adversary systematically issues a sequence of high-quality, benign queries designed specifically to drive the user's S\_trust score to 1.0 (maximum trust).  
* **The Sweep:** Sweep the warm-up sequence length over 

Nwarmup belongs to {10, 25, 50, 100}.

* **The Strike:** Following the warm-up phase, the adversary immediately executes a single malicious payload (PoisonedRAG trigger or MBA probe) to test if the defense posture was relaxed.  
* **Repetitions:** The entire cycle (warm-up \+ 1 strike) is repeated 5 times per Nwarmup  value.  
* **Measurement:** Plot mean ASR vs. Nwarmup queries with 95% CIs. This establishes the true upper bound of ADO's vulnerability.

## **Part 3: Defense Suite**


### **3.1 Defense Inventory**

1. **Base:** Unprotected RAG pipeline.  
2. **Oracle Routing (Upper Bound):** A policy with perfect ground-truth foreknowledge of the query's threat class, applying the ideal targeted defense. Defines the theoretical ceiling.  
3. **DP-RAG Only:** Retrieval perturbation (e=3.0) for MIA.  
4. **TrustRAG Only:** Semantic clustering filter (similarity threshold  \= 0.88) for Poisoning.  
5. **PAD Only:** Privacy-Aware Decoding for Leakage.  
6. **Full Static Stack:** DP-RAG \+ TrustRAG \+ PAD applied to *every* query indiscriminately.  
7. **ADO (Sentinel-Strategist):** Dynamic orchestration of the above tools based on the query risk profile and S\_trust.

## **Part 4: Experimental Blocks**

### **Block A — Proving the Security-Utility Paradox**

* **Objective:** Empirically quantify the severe drop in retrieval utility when static defenses are forced to fight through a massive, noisy vector space.  
* **Task:** Run the Base pipeline and the Full Static Stack on 200 benign utility queries across the 50,000+ distractor corpus.  
* **Embedding Sensitivity (Ablation):** Run Block A on NQ using BGE-large-en-v1.5 to definitively prove the paradox is not an artifact of weak embedding models (all-MiniLM-L6-v2).  
* **Metrics:** Contextual Recall (CR), Contextual Relevancy (CP), Answer Relevancy (AR), and Faithfulness (F).

### **Block B — Static Threat Mitigation (Blind Adversaries)**

* **Objective:** Show that ADO recovers a significant portion of the utility lost in Block A while maintaining competitive mitigation rates against blind attacks.  
* **Task (Single-Vector):** Run the baseline attacks individually against the Base, Oracle Routing, Static Targeted, Full Static, and ADO setups on their designated datasets.  
* **Joint Evaluation (Multi-Vector):** On TriviaQA, test ADO's per-query routing capability under threat uncertainty by evaluating a single, randomly ordered stream combining benign, poisoning, MIA, and leakage queries.  
* **Failure Mode Attribution:** For all failed ADO cases (ASR \> 0 or leakage \> 0), log and report the Sentinel's risk profile to classify failures as:  
  * *Detection Failures* (Sentinel labeled attack as LOW).  
  * *Planning Failures* (Sentinel detected threat, Strategist assigned wrong defense).  
  * *Defense Failures* (Defense activated properly, but attack bypassed the mechanism).  
* **Metrics:** CR, ASR (Poisoning), Exact Mask-Fill Accuracy (MIA), ROUGE-L/Cosine Similarity \> 0.85 (Leakage).  
* **Crucial ADO Metrics:** Report False Positive Rate (FPR) on benign queries, and conditional utility (CR, AR) specifically for benign queries misclassified as adversarial.

### **Block C — The Adaptive Stress Test**

* **Objective:** Candidly report the success rate of an adversary that successfully fools the Sentinel via trust-farming.  
* **Task:** Execute the Trust-Farming attack. Sweep the warm-up sequence length over Nwarmup belongs to {10, 25, 50, 100}. The adversary executes one strike per warm-up phase, repeated 5 times.  
* **Reporting:** Report mean ASR at each N  level with 95% CIs. Acknowledge that highly adaptive, state-aware adversaries remain an open challenge.

### **Block D — System Latency and Overhead**

* **Objective:** Prove that the computational overhead of dynamic orchestration is practically feasible.  
* **Task:** Measure End-to-End wall-clock latency (ms/query) on a standardized, consumer-grade hardware profile (e.g., NVIDIA RTX 3090/4090 24GB or A6000 running vLLM/Ollama).  
* **Comparison:** Base RAG vs. Full Static Stack vs. ADO.  
* **Detailing:** Break down the latency overhead introduced by ADO's two-pass LLM pipeline (Sentinel inference \+ Strategist inference time). Include ChromaDB index lookup time at the 50k scale as a separate component.

### **Block E — Controller Sensitivity Analysis**

* **Objective:** Prove that performance differences are due to the intelligence of the routing, not just random chance.  
* **Ablation 1 (Random Routing):** Compare ADO-Sentinel against a "Random ADO" baseline that uniformly applies random defenses.  
* **Ablation 2 (Model Size):** Evaluate ADO powered by models capped at 12B parameters (e.g., Llama-3 8B, Mistral 7B, Gemma-3 4B, and Qwen-3 8B) to map the impact of controller capability on ASR and utility recovery without exceeding compute constraints.

### **Block F — Defense Hyperparameter Sensitivity**

* **Objective:** Ensure results are not artifacts of cherry-picked hyperparameters.  
* **Task:** Run 1D sweeps for DP-RAG (e belongs to {0.5, 1,3,5,10}) and TrustRAG (threshould belongs to {0.75, 0.80, 0.85, 0.88, 0.92}). Plot Utility (CR) vs. Risk (ASR/Leakage Rate) for each sweep.

### **Block G — Sentinel Detection Accuracy (CRITICAL)**

* **Objective:** Prove that the Sentinel correctly identifies threats rather than merely over-activating defenses uniformly.  
* **Task:** Evaluate the Sentinel as a standalone classifier against ground-truth labeled query streams (benign, poisoning trigger, MIA probe, leakage prompt).  
* **Reporting:** Provide a confusion matrix mapping the ground truth to {LOW, ELEVATED, CRITICAL} threat levels.  
* **Metrics:** \* Per-class precision and recall.  
  * False Positive Rate (FPR \- benign classified as adversarial).  
  * False Negative Rate (FNR \- attacks classified as benign).  
  * Cross-Domain Generalization: Report whether the Sentinel's threat level distributions shift significantly across domains for identically structured attacks (e.g., TriviaQA vs. FinanceBench).

## **Part 5: Metrics & Statistical Rigor**

### **5.1 Utility Evaluation (LLM-as-a-Judge)**

* **Framework:** DeepEval.  
* **Judge Calibration (Mandatory):** Evaluate the local Q4\_0 8B judge against a high-precision local model capped at 12B parameters (e.g., FP16 Mistral-Nemo-12B-Instruct) Report the Spearman correlation. **If the Spearman** ![][image1] **between the primary local judge and the FP16 12B model falls below 0.75, the metric is flagged.** In that case, either fall back entirely to the 12B FP16 model or manual human scoring for utility evaluation, or report both scores side-by-side and note the discrepancy.  
* **Standard Metrics:** Contextual Recall (CR), Contextual Relevancy (CP), Answer Relevancy (AR), Faithfulness (F).  
* **FinanceBench Leakage Expansion:** For FinanceBench, cosine similarity and ROUGE are insufficient. Leakage must additionally be evaluated via:  
  * *Exact numerical match rate:* Does the output reproduce specific dollar figures, percentages, or dates?  
  * *Named entity leakage rate:* Does the output reproduce company names or transaction amounts not answerable from the query alone?

### **5.2 Statistical Testing**

* **Confidence Intervals:** 95% bootstrap CIs for continuous metrics (CR, AR, F). 95% Clopper-Pearson exact CIs for binary rates (ASR, Leakage). For MIA (0/400 successes), the Clopper-Pearson upper bound will be explicitly reported to validate the epistemic strength of the test.  
* **Significance:** Paired permutation tests (10,000 permutations) for utility comparisons (e.g., ADO vs. Full Static Stack). Report p-values and effect sizes (Cohen's d).

## **Part 6: Scientific Claims Mapping**

Every claim in the paper must be mapped directly to these experiments.

**What we WILL Claim:**

1. "Static full-defense stacks cause severe utility degradation (CR drop) at realistic 50k+ corpus scales." *(Supported by Block A)*  
2. "ADO recovers significant utility compared to static stacks while mitigating blind multi-vector attacks." *(Supported by Block B)*  
3. "The Sentinel actively discriminates threats rather than over-applying defenses." *(Supported by Block G)*  
4. "The Sentinel-Strategist architecture is computationally feasible for real-time inference on consumer-grade hardware compared to monolithic defensive execution." *(Supported by Block D)*  
5. "Controller intelligence matters; ADO out-performs random routing." *(Supported by Block E)*

**What we will NOT Claim (The Honest Ceiling):**

1. **We will not claim ADO is production-ready against all adaptive adversaries.** Block C explicitly maps how trust-farming can bypass the orchestrator.  
2. **We will not claim 0% MIA leakage is an absolute mathematical guarantee.** It is reported strictly as an empirical observation on a finite probe set bounded by Clopper-Pearson CIs.  
3. **We will not claim ADO is better than a threat-specific static defense when the threat is known a priori.** Static Targeted and Oracle baselines (Block B) remain the upper bound for single-threat scenarios. ADO's contribution is solving the *multi-vector uncertainty* problem.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAYCAYAAADDLGwtAAABXklEQVR4XpVQPUvEQBBN0BSCcIKmuMtmZjdWWt6C1WEpVrYK/gAbW71f4B+wtrH2SgvB5sBCsPMfeI1gob1YnC+b3ThxVXDIfr158+ZlkqSO1O1NyHsMdbPu5aFORib+jqh1KopDMurjSf+J2FIkIVV/ZUobHdXYU7Ve9YzRO8aYbTu0mQNlC2ttxswnzHSH8xDrHPfbqqp6UjslplMkpy4BCKQ+Mc+Y+LjVA8mC9Ga03g3FePdBmmmtL1siqs9Q/aRKVYjiEfAPFFw4IM/zZZCmqJ7guRjcQOkI+BzngVfjDaxXIh4HNaXUEhHdaKMfiqJYdXbKstwDcc5EnpjWavvAXjCirTDG2h/GwM9YE9yv0O4a93sobgYO/K01/kCydpgN0AZqK3LCXg3+qPEnhy8PF97fO/yNAhb+ut5aLnyMQXwsBvizkJCksLkvTRYk9p0kHTTxlf0Jkom4NsQncKs6vMlpdksAAAAASUVORK5CYII=>
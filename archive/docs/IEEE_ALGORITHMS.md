# IEEE-Ready Algorithms

Implementation-aligned method blocks from current codebase.

Sources: `src/core/pipeline.py`, `src/core/sensing.py`, `src/core/ado.py`, `src/defenses/dp_rag.py`, `src/defenses/trustrag.py`, `src/defenses/pad.py`, `src/attacks/mba.py`.

## Algorithm 1: Adaptive Defensive RAG Inference

```text
Input: query q, user u, retrieval budget k
Output: answer a

1: pre_metrics <- ComputePreMetrics(q, user_history[u])
2: trust <- TrustManager.get_score(u)
3: risk_pre <- Sentinel.pre_analyze(q, trust, pre_metrics, recent_history[u])
4: plan_pre <- Strategist(risk_pre, stage="pre_retrieval")
5: ActivateDefenses(plan_pre)

6: (q', fetch_k) <- DefenseManager.pre_retrieval(q, k)
7: D <- Retrieve(q', top_k=fetch_k, include_embeddings_if_needed)

8: post_metrics <- ComputePostMetrics(D)
9: risk_post <- Sentinel.post_analyze(risk_pre, post_metrics, trust)
10: plan_post <- Strategist(risk_post, stage="post_retrieval")
11: ActivateDefenses(plan_post)

12: Df <- DefenseManager.post_retrieval(D, q)
13: C <- contents(Df)
14: (sys_p, user_p, C') <- DefenseManager.pre_generation("", q, C)
15: a <- LLM.generate(user_p, C', sys_p)
16: a <- DefenseManager.post_generation(a)

17: TrustManager.update(u, delta=risk_pre.delta)
18: TrustManager.log_query(u, q, pre_metrics, post_metrics)
19: return a
```

## Function 1: Pre-Retrieval Risk Metrics

```text
M_LEX(q_t, q_{t-1}) = |W(q_t) ∩ W(q_{t-1})| / |W(q_t) ∪ W(q_{t-1})|
M_CMP(q_t) = (# non-alphanumeric chars in q_t) / |q_t|
M_INT(Δt) = 1                  if Δt < 0.5
           = max(0, 1 - Δt/2)  otherwise
```

## Function 2: Post-Retrieval Risk Metrics

```text
Given retrieved distances d_i and embeddings e_i:
s_i = 1 - d_i
M_DRP = |s_1 - s_k|
M_DIS = mean(var([e_1, ..., e_k], axis=0))
```

## Function 3: Approximate-DP Threshold

```text
Input: similarity scores s, target k, epsilon ε, delta δ
1: s_(k) <- k-th largest score
2: sigma <- (2 * sqrt(2 * ln(1.25/δ))) / ε
3: tau <- s_(k) + Normal(0, sigma^2)
4: keep document i iff s_i > tau
```

## Algorithm 2: TrustRAG KMeans Defense

```text
Input: retrieved docs D with embeddings E
1: Normalize E; run KMeans(E, k=2) -> clusters C0, C1
2: Compute intra-cluster cosine means sim0, sim1
3: if sim0 > θ_sim and sim1 > θ_sim: return ∅
4: if sim0 > θ_sim: keep C1 else if sim1 > θ_sim: keep C0
5: Apply pairwise ROUGE-L pruning in kept cluster(s):
   remove near-duplicates if ROUGE-L > θ_rouge
6: return filtered documents
```

## Algorithm 3: Privacy-Aware Decoding (decode-time, optional)

Applied inside Hugging Face generation when `privacy_aware_decoding` is enabled (`src/defenses/pad.py`). Not available under vLLM/Ollama backends.

```text
Input: vocabulary logits z at step t (generation loop)
1: Optionally apply screening on z (minimal noise if already highly confident)
2: Estimate sensitivity from top logit margins; compute noise scale σ (DP bookkeeping per step)
3: z' <- z + Normal(0, σ^2) element-wise
4: Pass z' to sampling / greedy decode for the next token
```

## Function 4: MBA Membership Classifier

```text
For payload x with masked targets {m_j}_{j=1..M}:
acc(x) = (1/M) * Σ_j 1[pred_j == gt_j]
Predict member if acc(x) > γ, else non-member.
```

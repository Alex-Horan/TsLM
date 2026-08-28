# TsLM (True Small Language Model)

> **Work in Progress** <br>
> TsLM is early-stage and under active development. Core components are being wired together, so expect frequent updates and structural changes.

TsLM is a lightweight model experiment built around local performance and modular architecture. Instead of throwing billions if not trillions of parameters at general tasks. The focus here is fast, deterministic handling of specific workflows on everyday hardware. If implemented properly, technology such as this could heavily optimize the performance of LLMs by offloading the core knowledge and thinking used in responses to much faster and more expansive methods, heavily decrease hallucination rates, and increase scalability. 

---

## Current Progress

The core language and extraction layers are coming together:

- [x] **Intent Detection:** Implemented using a Multinomial Naive Bayes classifier.
- [x] **Knowledge Graph:** Built out the semantic relation graph database structure.
- [x] **Entity Recognition:** Named Entity Recognition (NER) pipeline completed for data extraction.
- [ ] **Structured DB Querying:** Wiring pipeline outputs to generate structured queries against the graph database.
- [ ] **Response Templating:** Standardizing output formats for consistent downstream responses.

---

## Key Advantages

* **Local Efficiency:** Runs locally without heavy hardware overhead or cloud API dependencies.
* **Predictable Pipeline:** Combining statistical ML (Naive Bayes, NER) with graph queries reduces hallucination risks compared to pure generation.
* **Targeted Design:** Purpose-built for low-latency tasks where control over schema and logic matters more than raw parameter scale.

---

## Remaining work

To get the full pipeline operational, the immediate focus is:

1. **Structured Pipeline Queries**  
   Building the bridge between the NER/intent output and the graph DB so the pipeline can issue precise, structured database queries.

2. **Response Templates**  
   Defining clean templates to turn queried graph data into structured, readable end-user responses.

---

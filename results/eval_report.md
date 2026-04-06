# RAG System Evaluation Report

**Date:** 2026-04-04  
**Test Cases:** 10  
**Runs per Case:** 1

## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Context Precision | 1.000 | ±0.000 |
| Answer Similarity | 0.883 | ±0.062 |
| Retrieval Latency (ms) | 22.752 | ±19.600 |
| Generation Latency (ms) | 969.872 | ±297.585 |
| End-to-End Latency (ms) | 992.624 | ±297.668 |
| Indexing Latency (ms) | 166.758 | ±65.950 |
| Time to First Token (ms) | 0.000 | ±0.000 |
| Tokens/sec (approx.) | 56.983 | ±34.686 |
| Routing Accuracy | 1.000 | ±0.000 |
| Faithfulness Rate | 1.000 | ±0.000 |

## Retrieval Metrics (per Test Case)

| ID | Description | Chunks | Context Precision | Avg Distance | Retrieval Latency (ms) |
|----|-------------|--------|-------------------|--------------|------------------------|
| TC-001 | Small document — basic factual recall | 3.00 | 1.00 | 0.74 | 62.98 |
| TC-002 | Small document — conceptual understandin | 3.00 | 1.00 | 0.81 | 56.51 |
| TC-003 | Medium document — specific detail retrie | 4.00 | 1.00 | 0.85 | 11.14 |
| TC-004 | Medium document — deep technical detail | 4.00 | 1.00 | 1.11 | 14.62 |
| TC-005 | Large document — broad topic comprehensi | 4.00 | 1.00 | 0.86 | 15.59 |
| TC-006 | Large document — numerical data retrieva | 4.00 | 1.00 | 1.02 | 14.93 |
| TC-007 | Large document — multi-section synthesis | 4.00 | 1.00 | 0.77 | 13.61 |
| TC-008 | CSV spreadsheet — specific data lookup | 4.00 | 1.00 | 0.86 | 12.54 |
| TC-009 | CSV spreadsheet — cross-row comparison | 4.00 | 1.00 | 0.80 | 12.75 |
| TC-010 | Large document — specific statistic from | 4.00 | 1.00 | 0.83 | 12.86 |

## Generation Metrics (per Test Case)

| ID | Faithfulness | Answer Similarity | Gen Latency (ms) | TTFT (ms) | Tokens/sec |
|----|-------------|-------------------|-------------------|-----------|------------|
| TC-001 | 1.00 | 0.91 | 1225.51 | 0.00 | 16.32 |
| TC-002 | 1.00 | 0.91 | 688.32 | 0.00 | 126.39 |
| TC-003 | 1.00 | 0.81 | 1389.27 | 0.00 | 95.73 |
| TC-004 | 1.00 | 0.84 | 754.01 | 0.00 | 71.62 |
| TC-005 | 1.00 | 0.92 | 639.12 | 0.00 | 26.60 |
| TC-006 | 1.00 | 0.83 | 766.39 | 0.00 | 56.11 |
| TC-007 | 1.00 | 0.82 | 1415.67 | 0.00 | 50.86 |
| TC-008 | 1.00 | 0.94 | 726.26 | 0.00 | 22.03 |
| TC-009 | 1.00 | 0.85 | 1114.49 | 0.00 | 66.40 |
| TC-010 | 1.00 | 1.00 | 979.68 | 0.00 | 37.77 |

## System Metrics (per Test Case)

| ID | Routing Correct | Indexing (ms) | E2E Latency (ms) | LLM Source |
|----|-----------------|---------------|-------------------|------------|
| TC-001 | 1.00 | 127.49 | 1288.49 | Groq |
| TC-002 | 1.00 | 79.56 | 744.83 | Groq |
| TC-003 | 1.00 | 231.57 | 1400.41 | Groq |
| TC-004 | 1.00 | 112.02 | 768.62 | Groq |
| TC-005 | 1.00 | 221.47 | 654.71 | Groq |
| TC-006 | 1.00 | 297.34 | 781.32 | Groq |
| TC-007 | 1.00 | 147.94 | 1429.27 | Groq |
| TC-008 | 1.00 | 147.69 | 738.80 | Groq |
| TC-009 | 1.00 | 123.34 | 1127.24 | Groq |
| TC-010 | 1.00 | 179.16 | 992.54 | Groq |

## Sample Generated Answers

### TC-001: What is the chemical equation for photosynthesis?

**Expected:** The overall chemical equation for photosynthesis is 6CO2 + 6H2O + light energy yields C6H12O6 + 6O2. Six molecules of carbon dioxide and six molecules of water, using light energy, produce one molecul...

**Generated:** The overall chemical equation for photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2 (Found from document)...

### TC-002: What are the two main stages of photosynthesis and where do they occur?

**Expected:** The two main stages are the light-dependent reactions, which take place in the thylakoid membranes, and the light-independent reactions (Calvin cycle), which occur in the stroma of the chloroplast....

**Generated:** The two main stages of photosynthesis are the light-dependent reactions and the light-independent reactions, also known as the Calvin cycle. The light-dependent reactions take place in the thylakoid membranes of the chloroplast, where chlorophyll absorbs sunlight and uses it to split water molecules...

### TC-003: What are the three primary types of machine learning?

**Expected:** The three primary types of machine learning are supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data, unsupervised learning finds patterns in u...

**Generated:** The three primary types of machine learning are: 
1. Supervised Learning: The algorithm is trained on labeled data, where each input has a corresponding correct output. The model learns to map inputs to outputs. 
2. Unsupervised Learning: 
3. (Although the provided context does not explicitly state ...

### TC-004: What are transformers in deep learning and when were they introduced?

**Expected:** Transformers were introduced in the 2017 paper 'Attention Is All You Need.' They use self-attention mechanisms to process all positions in a sequence simultaneously. Transformers form the foundation o...

**Generated:** Transformers in deep learning are a type of neural network architecture that uses self-attention mechanisms to process all positions in a sequence simultaneously. They were introduced in the 2017 paper "Attention Is All You Need" and have since formed the foundation of modern large language models l...

### TC-005: How much has the global average temperature risen above pre-industrial levels?

**Expected:** The global average temperature has risen by approximately 1.1 degrees Celsius above pre-industrial levels as of 2023, and the rate of warming is accelerating....

**Generated:** The global average temperature has risen by approximately 1.1 degrees Celsius above pre-industrial levels. (Found from document)...

### TC-006: What is the current CO2 concentration in the atmosphere compared to pre-industrial times?

**Expected:** Carbon dioxide levels have risen from approximately 280 parts per million (ppm) in pre-industrial times to over 420 ppm in 2024. Methane concentrations have also more than doubled since 1750....

**Generated:** The current CO2 concentration in the atmosphere is over 420 parts per million (ppm) in 2024, whereas in pre-industrial times it was approximately 280 parts per million (ppm). This represents a significant increase in CO2 levels due to human activities. (Found from document)...

### TC-007: What are the key mitigation strategies for climate change discussed in the document?

**Expected:** The key mitigation strategies include energy transition from fossil fuels to renewable sources like solar and wind, transportation electrification with electric vehicles, carbon capture and storage te...

**Generated:** The key mitigation strategies for climate change discussed include reducing greenhouse gas emissions, with a focus on an energy transition from fossil fuels to renewable energy sources such as solar and wind power, as well as the use of nuclear energy for reliable, low-carbon baseload power. Additio...

### TC-008: What was the revenue for Widget B in North America in Q4 2024?

**Expected:** The revenue for Widget B in North America in Q4 2024 was $187,500 with 1,250 units sold and a profit margin of 33.1%....

**Generated:** The revenue for Widget B in North America in Q4 2024 was $187,500. (Found from document)...

### TC-009: Which region had the highest profit margin for Widget A?

**Expected:** North America had the highest profit margin for Widget A, reaching 26.8% in Q4 2024, compared to Europe's peak of 23.2% and Asia Pacific's peak of 22.3%....

**Generated:** To find the region with the highest profit margin for Widget A, we can look at the provided context. There is only one mention of Widget A, which is in the context for Europe, with a profit margin of 23.2%. Since there is no comparison available for other regions for Widget A, we conclude that Europ...

### TC-010: What does the Paris Agreement aim to achieve regarding global temperature?

**Expected:** The Paris Agreement aims to hold the increase in global average temperature to well below 2 degrees Celsius above pre-industrial levels and to pursue efforts to limit the temperature increase to 1.5 d...

**Generated:** The Paris Agreement aims to hold the increase in global average temperature to well below 2 degrees Celsius above pre-industrial levels and to pursue efforts to limit the temperature increase to 1.5 degrees Celsius. (Found from document)...

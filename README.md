
# Clinical Trials failed due to Negative Efficacy

Identifying the trials failed due to poor efficacy. This aims to comprehensively identify wrong druggability hypotheses based which progressed to Phase 2/3 where relevant endpoints have not been met.


## Original work: [Why clinical trials are terminated Pak et al 2015](https://www.biorxiv.org/content/10.1101/021543v1)

- Clinical Trails from NCT Archive, with status of Terminated, Withdrawn, Suspended
- Manually curated 3124 reasons for termination
   - a short text e.g. "Hiatus for administrative reasons." for NCT00641810 or "DSMB" for NCT00536770 (DSMB stands for Data and Safety Monitoring Board).
- Up to 3 reasons for suspension / termination based on their ontology of 38 terms
- 6 Top-level categories
 	- Negative efficacity
 	- Positive
 	- Negative safety
 	- Possibly negative
 	- Neutral
 	- Misuse

## OpenTargets work by [Razuvayevskaya et al. 2022](https://www.nature.com/articles/s41588-024-01854-z#Sec19)
- Added 447 Studies terminated due to COVID-19 Pandemic
- Grouped / collapsed 23 of the original categories to 7 revised categories based on cosine similarity between embeddings
- Fine-tuned BERT model into stop reasons, achieving F_micro=0.91 on CV, test split included 407 datapoints - unclear split method.
- Uses multi-label prediction, for Negative category specifically - pF1=0.92, support=32
- Human annotation on 1675 additional [ClinicalTrials.gov](ClinicalTrials.gov) studies not in the previous applications, marked by 7 annotators - withheld for evaluation - (data not available).
- On the additional dataset F_micro was between 0.7-0.83 depending on the annotator.
- Fine-tuned the model with the additional curated data.
- Applied to 28,561 trials stopped before 27 Nov 2021
- Used the model to assess the correlation between the stop results and available genetic evidence for a given indication (based on indication-approved drugs and their mechanism of action to targets based on ChEMBL).
  
- Total dataset provided: 3,747 studies, 368 (9.82%) are assigned Negative label:
   - of which 64 assigned to more labels
      - reviewed the labels
       - co-labels to exclude: Business_Administrative, Ethical_Reason, Insufficient_Enrollment, Regulatory, Safety_Sideeeffects, Study_Staff_Moved, Success
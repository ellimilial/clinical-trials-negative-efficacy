Obtained 18/04/2026 at 17:30 from https://aact.ctti-clinicaltrials.org

Query:
```
SELECT DISTINCT
  ctgov.studies.nct_id,
  ctgov.studies.brief_title,
  ctgov.studies.official_title,
  ctgov.studies.overall_status,
  ctgov.studies.start_date,
  ctgov.studies.completion_date,
  ctgov.studies.study_type,
  ctgov.studies.phase,
  ctgov.studies.why_stopped,
  ctgov.conditions.name as condition_name,
  ctgov.interventions.name as intervention_name,
  ctgov.interventions.intervention_type as intervention_type
FROM ctgov.studies
LEFT JOIN ctgov.interventions
  ON ctgov.interventions.nct_id = ctgov.studies.nct_id
LEFT JOIN ctgov.conditions ON ctgov.conditions.nct_id = ctgov.studies.nct_id
```

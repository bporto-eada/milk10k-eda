# Session 2 findings re-checked on the full dataset

Produced by `python milestone1.py eda`.

| Session 2 finding | Session 2 | full dataset | still true? | consequence for the pipeline |
|---|---|---|---|---|
| Malignant images are redder than Benign ones | mean R 177.5 Malignant vs 165.1 Benign (72 images) | mean R 174.2 vs 170.7 (all 10,480 images) | yes | colour carries class signal: keep RGB input and never jitter hue (A3.5) |
| Image type shifts colour more than the class does | mean B gap: image type 16.7 vs class 7.1 (72 images) | mean B gap: type 31.2 vs class 6.8 (all images) | yes | both views share one model: normalise with train stats over both types, evaluate per lesion |
| image_manipulation ('altered') is linked to the diagnosis | Cramer's V 0.139 | Cramer's V 0.139 | yes | possible shortcut: never a model input; report test results for altered images separately |
| diagnosis_confirm_type / concomitant_biopsy predict the label | Cramer's V 0.311 | Cramer's V 0.311 | yes | label leakage: excluded from the inputs (data_quality_report.md) |
| anatom_site_general is often missing | 37.3% missing | 37.3% missing; NV share +11.0 pp when missing | yes | missing is informative (mostly trunk): if site is used, encode 'unknown', never impute the mode |
| Indeterminate and Malignant patients are older | median 55 / 70 / 65 (Benign / Indet. / Malignant) | median 55 / 70 / 65 | yes | age is a legitimate clinical input for a later metadata branch; impute 0.4% missing with train median |
| image_type says nothing about the label | Cramer's V 0.000 | Cramer's V 0.000 | yes | not a shortcut: every lesion has one image of each type, so every class has both views |

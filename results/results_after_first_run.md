
======================================================================
RUNNING ALL EXPERIMENTS FOR: GENDER
======================================================================

--------------------------------------------------
PHASE 1: BASELINES (E1)
--------------------------------------------------
2026-01-07 01:25:35.185902: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-01-07 01:25:35.203451: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1767749135.225413    3266 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1767749135.231974    3266 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1767749135.248460    3266 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1767749135.248490    3266 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1767749135.248493    3266 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1767749135.248496    3266 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
2026-01-07 01:25:35.253397: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

[E1.1] Majority baseline...

============================================================
E1.1: MAJORITY BASELINE - GENDER
============================================================

Data loaded:
  Train: 7000 samples
  Dev:   1500 samples
  Test:  1500 samples

Majority class: '1' (3500/7000 = 50.0%)
Label distribution in train:
  0: 3500 (50.0%)
  1: 3500 (50.0%)

==================================================
Dev Set Metrics
==================================================
  Accuracy:    0.5000
  Macro F1:    0.3333
  Weighted F1: 0.3333
  Precision:   0.2500
  Recall:      0.5000

  Per-class F1:
    0: 0.0000 (n=750.0)
    1: 0.6667 (n=750.0)

==================================================
Test Set Metrics
==================================================
  Accuracy:    0.5000
  Macro F1:    0.3333
  Weighted F1: 0.3333
  Precision:   0.2500
  Recall:      0.5000

  Per-class F1:
    0: 0.0000 (n=750.0)
    1: 0.6667 (n=750.0)
Results saved to: /content/lang-and-ai/results/E1_baselines/gender/e1_1_majority_results.json

[E1.2] Keyword baseline...

============================================================
E1.2: KEYWORD HEURISTIC - GENDER
============================================================

Data loaded:
  Train: 7000 samples
  Dev:   1500 samples
  Test:  1500 samples

Keyword matching coverage:
  Dev:  743/1500 (49.5%)
  Test: 743/1500 (49.5%)

==================================================
Dev Set Metrics
==================================================
  Accuracy:    0.3100
  Macro F1:    0.2656
  Weighted F1: 0.2656
  Precision:   0.2493
  Recall:      0.3100

  Per-class F1:
    0: 0.0849 (n=750.0)
    1: 0.4462 (n=750.0)

==================================================
Test Set Metrics
==================================================
  Accuracy:    0.2980
  Macro F1:    0.2588
  Weighted F1: 0.2588
  Precision:   0.2438
  Recall:      0.2980

  Per-class F1:
    0: 0.0883 (n=750.0)
    1: 0.4293 (n=750.0)

Matched samples accuracy: 123/743 = 16.6%
Results saved to: /content/lang-and-ai/results/E1_baselines/gender/e1_2_keyword_results.json

[E1.3] SVM baseline (polluted)...

============================================================
E1.3: STYLOMETRIC SVM - GENDER (polluted)
============================================================

Data loaded from: /content/lang-and-ai/experiments/E1_baselines/gender
  Train: 7000 samples
  Dev:   1500 samples
  Test:  1500 samples
  Classes: ['0', '1']

Training SVM...
  Config: {'ngram_range': (3, 5), 'max_features': 50000, 'analyzer': 'char_wb', 'C': 1.0}
Evaluating...

==================================================
Dev Set Metrics
==================================================
  Accuracy:    0.9000
  Macro F1:    0.9000
  Weighted F1: 0.9000
  Precision:   0.9000
  Recall:      0.9000

  Per-class F1:
    0: 0.9004 (n=750.0)
    1: 0.8996 (n=750.0)

==================================================
Test Set Metrics
==================================================
  Accuracy:    0.8927
  Macro F1:    0.8927
  Weighted F1: 0.8927
  Precision:   0.8927
  Recall:      0.8927

  Per-class F1:
    0: 0.8925 (n=750.0)
    1: 0.8929 (n=750.0)

Top features for class '1':
  'f) ': 1.5155
  '‍♀️': 1.3552
  '‍♀️ ': 1.2440
  '♀️ ': 1.2241
  ' he ': 1.1515
  'husba': 1.1277
  'husb': 1.1277
  'usban': 1.1231
  'usba': 1.1001
  'sband': 1.0986

Top features for class '0':
  'wife ': -2.3074
  'm) ': -2.2481
  ' wife': -2.1349
  'wife': -2.1003
  ' wif': -2.0973
  'wif': -2.0649
  ' gay': -1.4250
  'gay': -1.3666
  '‍♂️': -1.3525
  '‍♂️ ': -1.2689

Model saved to: /content/lang-and-ai/models/E1_baselines/gender/svm_polluted.joblib
Results saved to: /content/lang-and-ai/results/E1_baselines/gender/e1_3_svm_polluted_results.json

[E1.4] RoBERTa baseline (polluted)...

============================================================
E1.4: RoBERTa-BASE - GENDER (polluted)
============================================================
Config: {'learning_rate': 2e-05, 'batch_size': 16, 'epochs': 3, 'max_length': 128, 'warmup_ratio': 0.1, 'weight_decay': 0.01}

Data loaded from: /content/lang-and-ai/experiments/E1_baselines/gender
  Train: 7000 samples
  Dev:   1500 samples
  Test:  1500 samples
  Label map: {'0': 0, '1': 1}

Training RoBERTa...

Using device: cuda
tokenizer_config.json: 100% 25.0/25.0 [00:00<00:00, 61.3kB/s]
vocab.json: 100% 899k/899k [00:00<00:00, 11.7MB/s]
merges.txt: 100% 456k/456k [00:00<00:00, 19.8MB/s]
tokenizer.json: 100% 1.36M/1.36M [00:00<00:00, 14.7MB/s]
config.json: 100% 481/481 [00:00<00:00, 4.07MB/s]
model.safetensors: 100% 499M/499M [00:00<00:00, 522MB/s]
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

Epoch 1/3
Training: 100% 438/438 [01:43<00:00,  4.23it/s]
  Train loss: 0.6054
Evaluating: 100% 94/94 [00:19<00:00,  4.75it/s]
  Dev Accuracy: 0.7153
  Dev Macro-F1: 0.7095

Epoch 2/3
Training: 100% 438/438 [01:34<00:00,  4.66it/s]
  Train loss: 0.4537
Evaluating: 100% 94/94 [00:18<00:00,  5.03it/s]
  Dev Accuracy: 0.7507
  Dev Macro-F1: 0.7491

Epoch 3/3
Training: 100% 438/438 [01:35<00:00,  4.59it/s]
  Train loss: 0.3244
Evaluating: 100% 94/94 [00:18<00:00,  4.96it/s]
  Dev Accuracy: 0.7587
  Dev Macro-F1: 0.7583
Evaluating: 100% 94/94 [00:18<00:00,  5.00it/s]
Evaluating: 100% 94/94 [00:19<00:00,  4.73it/s]

==================================================
Dev Set Metrics (Final)
==================================================
  Accuracy:    0.7587
  Macro F1:    0.7583
  Weighted F1: 0.7583
  Precision:   0.7602
  Recall:      0.7587

  Per-class F1:
    0: 0.7490 (n=750.0)
    1: 0.7677 (n=750.0)

==================================================
Test Set Metrics
==================================================
  Accuracy:    0.7740
  Macro F1:    0.7738
  Weighted F1: 0.7738
  Precision:   0.7750
  Recall:      0.7740

  Per-class F1:
    0: 0.7670 (n=750.0)
    1: 0.7806 (n=750.0)

Model saved to: /content/lang-and-ai/models/E1_baselines/gender/roberta_polluted
Results saved to: /content/lang-and-ai/results/E1_baselines/gender/e1_4_roberta_polluted_results.json

--------------------------------------------------
PHASE 2: HYPOTHESIS TESTING (E2)
--------------------------------------------------

[E2.1] SVM on cleaned...

============================================================
E2.1: SVM ON CLEANED - GENDER
============================================================

Cleaned data from: /content/lang-and-ai/experiments/E2_hypothesis/gender/cleaned
  Train: 7000, Dev: 1500, Test: 1500

Training SVM on cleaned data...

==================================================
Dev Set (Cleaned)
==================================================
  Accuracy:    0.8953
  Macro F1:    0.8953
  Weighted F1: 0.8953
  Precision:   0.8953
  Recall:      0.8953

  Per-class F1:
    0: 0.8955 (n=750.0)
    1: 0.8951 (n=750.0)

==================================================
PERFORMANCE DROP ANALYSIS (H2)
==================================================
  Polluted F1: 0.9000
  Cleaned F1:  0.8953
  Absolute drop: 0.0047
  Relative drop: 0.5%

Model saved to: /content/lang-and-ai/models/E2_hypothesis/gender/svm_cleaned.joblib
Results saved to: /content/lang-and-ai/results/E2_hypothesis/gender/e2_1_svm_cleaned_results.json

[E2.2] RoBERTa on cleaned...

============================================================
E2.2: RoBERTa ON CLEANED - GENDER
============================================================

Cleaned data: Train=7000, Dev=1500, Test=1500

Training RoBERTa on cleaned data...

Using device: cuda
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

Epoch 1/3
Training: 100% 438/438 [01:43<00:00,  4.24it/s]
  Train loss: 0.6069
Evaluating: 100% 94/94 [00:19<00:00,  4.73it/s]
  Dev Accuracy: 0.7087
  Dev Macro-F1: 0.7050

Epoch 2/3
Training: 100% 438/438 [01:35<00:00,  4.57it/s]
  Train loss: 0.4684
Evaluating: 100% 94/94 [00:18<00:00,  4.96it/s]
  Dev Accuracy: 0.7333
  Dev Macro-F1: 0.7299

Epoch 3/3
Training: 100% 438/438 [01:36<00:00,  4.56it/s]
  Train loss: 0.3503
Evaluating: 100% 94/94 [00:18<00:00,  4.95it/s]
  Dev Accuracy: 0.7367
  Dev Macro-F1: 0.7363
Evaluating: 100% 94/94 [00:18<00:00,  4.97it/s]

==================================================
Dev Set (Cleaned)
==================================================
  Accuracy:    0.7367
  Macro F1:    0.7363
  Weighted F1: 0.7363
  Precision:   0.7379
  Recall:      0.7367

  Per-class F1:
    0: 0.7270 (n=750.0)
    1: 0.7457 (n=750.0)

==================================================
PERFORMANCE DROP ANALYSIS (H1)
==================================================
  Polluted F1: 0.7583
  Cleaned F1:  0.7363
  Absolute drop: 0.0220
  Relative drop: 2.9%
  ✗ H1 NOT SUPPORTED: Drop < 10 percentage points

Model saved to: /content/lang-and-ai/models/E2_hypothesis/gender/roberta_cleaned
Results saved to: /content/lang-and-ai/results/E2_hypothesis/gender/e2_2_roberta_cleaned_results.json

[E2.3] Cross-condition SVM...

============================================================
E2.3: CROSS-CONDITION SVM - GENDER
============================================================
Loaded model from: /content/lang-and-ai/models/E1_baselines/gender/svm_polluted.joblib

==================================================
Polluted→Cleaned (SVM)
==================================================
  Accuracy:    0.8900
  Macro F1:    0.8900
  Weighted F1: 0.8900
  Precision:   0.8900
  Recall:      0.8900

  Per-class F1:
    0: 0.8896 (n=750.0)
    1: 0.8904 (n=750.0)
Results saved to: /content/lang-and-ai/results/E2_hypothesis/gender/e2_3_svm_cross_results.json

[E2.4] Cross-condition RoBERTa...

============================================================
E2.4: CROSS-CONDITION RoBERTa - GENDER
============================================================
Evaluating: 100% 94/94 [00:21<00:00,  4.34it/s]

==================================================
Polluted→Cleaned (RoBERTa)
==================================================
  Accuracy:    0.7480
  Macro F1:    0.7475
  Weighted F1: 0.7475
  Precision:   0.7502
  Recall:      0.7480

  Per-class F1:
    0: 0.7357 (n=750.0)
    1: 0.7592 (n=750.0)
Results saved to: /content/lang-and-ai/results/E2_hypothesis/gender/e2_4_roberta_cross_results.json

--------------------------------------------------
PHASE 4: FINAL EVALUATION (E4)
--------------------------------------------------

[E4.1] Final SVM evaluation...

============================================================
E4: FINAL SVM - GENDER
============================================================

==================================================
Test Set (polluted)
==================================================
  Accuracy:    0.8927
  Macro F1:    0.8927
  Weighted F1: 0.8927
  Precision:   0.8927
  Recall:      0.8927

  Per-class F1:
    0: 0.8925 (n=750.0)
    1: 0.8929 (n=750.0)

==================================================
Test Set (cleaned)
==================================================
  Accuracy:    0.8907
  Macro F1:    0.8907
  Weighted F1: 0.8907
  Precision:   0.8907
  Recall:      0.8907

  Per-class F1:
    0: 0.8908 (n=750.0)
    1: 0.8905 (n=750.0)

  Performance drop: 0.2%
Results saved to: /content/lang-and-ai/results/E4_final/gender/e4_svm_final_results.json

[E4.2] Final RoBERTa evaluation...

============================================================
E4: FINAL RoBERTa - GENDER
============================================================
Evaluating: 100% 94/94 [00:21<00:00,  4.31it/s]

==================================================
Test Set (polluted)
==================================================
  Accuracy:    0.7740
  Macro F1:    0.7738
  Weighted F1: 0.7738
  Precision:   0.7750
  Recall:      0.7740

  Per-class F1:
    0: 0.7670 (n=750.0)
    1: 0.7806 (n=750.0)
Evaluating: 100% 94/94 [00:21<00:00,  4.29it/s]

==================================================
Test Set (cleaned)
==================================================
  Accuracy:    0.7567
  Macro F1:    0.7565
  Weighted F1: 0.7565
  Precision:   0.7572
  Recall:      0.7567

  Per-class F1:
    0: 0.7512 (n=750.0)
    1: 0.7619 (n=750.0)

  Performance drop: 2.2%
Results saved to: /content/lang-and-ai/results/E4_final/gender/e4_roberta_final_results.json

======================================================================
ALL EXPERIMENTS COMPLETE FOR: GENDER
Results saved to: /content/lang-and-ai/results
Models saved to: /content/lang-and-ai/models
======================================================================

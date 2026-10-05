"""Frozen settings used by the paper experiments."""
PROTOCOLS = {'adaptive_replication/protocol.json': {'id': 'count-adaptive-real-frozen-validation-20261003-v1',
                                        'declared_client_date': '2026-10-03 Asia/Shanghai',
                                        'status': 'Frozen before new real-data decision '
                                                  'computations; prior complete labels, earlier '
                                                  'results, and supplied simulator design evidence '
                                                  'were already known.',
                                        'primary': 'EV-count (exact analytic j/(m+1), no score '
                                                   'model)',
                                        'secondary': ['TargetK-cap',
                                                      'EV-score (frozen failure probabilities)',
                                                      'EV-flat-attachment (faithful empirical flat '
                                                      'approximation, diagnostic only)'],
                                        'settings': [[0.05, 0.1], [0.05, 0.05], [0.1, 0.1]],
                                        'repetitions': 500,
                                        'cap_fraction': 0.3,
                                        'source_K_grid': 'm_min_binomial * [1,2,4,8]; largest with '
                                                         'K/(source_rate*N)<=.2, else minimum; '
                                                         'released scorer no prior uses minimum',
                                        'EV_schedule': 'binomial rank increments over '
                                                       'm=1..round(.3N); use only observed n,m; '
                                                       'phat=(m+1)/(n+2); compare current proxy to '
                                                       'all future increments predicted within '
                                                       'cap; certify observed counts at cap',
                                        'EV_count_proxy': '(1-n/N)*j/(m+1) exactly',
                                        'EV_score_proxy': 'weighted empirical score quantiles '
                                                          'using fixed p=s; grid4001; beta '
                                                          'quantile quadrature64; strict lower '
                                                          'CDF; all-zero probability sum uses '
                                                          'analytic count proxy',
                                        'comparators': ['TargetK-fallback',
                                                        'TargetK-cap',
                                                        'TargetK-cap-binomial',
                                                        'Fixed20',
                                                        'EV-count',
                                                        'EV-score',
                                                        'EV-flat-attachment',
                                                        'EV-count-binomial',
                                                        'Count-checkpoints',
                                                        'K-grid-cap (4 K values)',
                                                        'Score-checkpoints-split'],
                                        'count_checkpoints': 'fractions .05,.1,.2,.3; stop at '
                                                             'first positive FPC rank, use full '
                                                             'delta; no sampled scores used for '
                                                             'stopping',
                                        'score_checkpoints': 'same fractions, delta/4; stop when '
                                                             'rank>0 and skip>=.1 or final look',
                                        'seed': '2026091100+10000*target_id+rep, paired with the '
                                                'original study',
                                        'tasks': 'all 75 frozen scorer/transfer/seed tasks; all 3 '
                                                 'settings; main HGB seed20260905 retained; no new '
                                                 'fitting or simulation',
                                        'aggregation': 'DS/SD equal-weight transfers; OP stable '
                                                       'and changed separate; duplicate source '
                                                       'v0.9.8->v0.10.0 excluded from summaries '
                                                       'but raw retained; fitting seeds reported '
                                                       'separately',
                                        'statistics': 'paired t intervals over per-repetition '
                                                      'transfer means; exact Clopper-Pearson '
                                                      'per-design violation intervals, exploratory '
                                                      'unadjusted checks; hindsight K grid '
                                                      'reference only',
                                        'validation': 'implementation and stopping invariance '
                                                      'first; exact small-population count-path '
                                                      'coverage; attached implementation '
                                                      'crosschecks; main anchors replicate old '
                                                      'Target-K and Fixed20, cap-certify equals '
                                                      'Fixed30 on source RL rare targets; no gain '
                                                      'tuning',
                                        'design_disclosure': 'EV designed on a simulator fitted to '
                                                             'published fixed-fraction results. '
                                                             'True current-library labels and '
                                                             'prior outcomes were already '
                                                             'available; this is a declared '
                                                             'retrospective validation, not '
                                                             'independent holdout or prospective '
                                                             'preregistration.',
                                        'no_manuscript_changes': True,
                                        'no_new_datasets': True,
                                        'no_new_simulator_executions': True},
 'replication/protocol.json': {'id': 'tse-revision-experiments-20261002-v1',
                               'status': 'Frozen before new revision experiments; original results '
                                         'and earlier audit were already known.',
                               'source': '../reproducibility',
                               'repetitions': 500,
                               'seed_base': 2026091100,
                               'target_seed_rule': 'seed_base+10000*target_id+rep; identical to '
                                                   'original protocol',
                               'fractions': [0.05, 0.1, 0.2, 0.3],
                               'epsilon': 0.05,
                               'delta': 0.1,
                               'risk_sensitivity': 'primary HGB only; n=.2N; '
                                                   'epsilon=.01,.02,.05,.1 at delta=.1, '
                                                   'delta=.01,.05,.1 at epsilon=.05',
                               'fixed_comparisons': 'FPC and Binomial for every frozen HGB split, '
                                                    'configuration HGB, logistic and released '
                                                    'ITS4SDC scorer',
                               'inverse_K': [45, 90, 180, 360],
                               'inverse_cap': 0.3,
                               'inverse_primary': 'largest K with K/(source_failure_rate*N)<=.2; '
                                                  'if none K=45; released ITS4SDC without source '
                                                  'labels uses predefined K=45',
                               'inverse_grid': 'all four K values and both FPC/Binomial rules for '
                                               'primary HGB, released ITS4SDC, and openpilot '
                                               'logistic; source-selected K for other scorers',
                               'adaptive': 'checkpoints .05,.1,.2,.3; delta/4; stop when j>0 and '
                                           'net skip>=.1',
                               'budget_B': [0.3, 0.5, 0.7],
                               'budget_n': [0.05, 0.1],
                               'budget_scopes': 'primary HGB, full-library ITS4SDC and openpilot '
                                                'logistic',
                               'budget_bound': 'conservative grid with retained lower feasibility '
                                               'constraint; also original reference grid; exact '
                                               'bound on 20 predeclared draws per task '
                                               '(rep=0,25,...475) for all six budget designs',
                               'planning': 'true-M analytic validation separately from source-rate '
                                           'prediction; availability probability targets .5 and '
                                           '.9; sample-size grid .005..995',
                               'family_analysis': 'all four openpilot scenario families; '
                                                  'family-specific miss; optional FPC delta/4 on '
                                                  'each family',
                               'aggregation': 'DS and SD equal-weight transfers; OP stable and '
                                              'changed cases separately; duplicate changed-source '
                                              'rows retained raw and excluded from case '
                                              'aggregation',
                               'cost': 'distinct executions including calibration; duration and '
                                       'decision overhead recorded separately; no new simulator '
                                       'runs',
                               'validation': 'compare every baseline draw with archived originals; '
                                             'exact binomial violation intervals; investigate CI '
                                             'lower bound>delta, do not infer invalidity from '
                                             'point frequency alone'},
 'reproducibility/protocol_0910.json': {'id': 'empirical-triage-0910-v1',
                                        'declared_date': '2026-09-11',
                                        'status': 'Follow-up protocol after earlier results were '
                                                  'known; frozen before this rerun.',
                                        'epsilon': 0.05,
                                        'delta': 0.1,
                                        'fractions': [0.05, 0.1, 0.2, 0.3],
                                        'epsilon_sensitivity': [0.01, 0.02, 0.05, 0.1],
                                        'delta_sensitivity': [0.01, 0.05, 0.1],
                                        'budget_fractions': [0.05,
                                                             0.1,
                                                             0.2,
                                                             0.3,
                                                             0.4,
                                                             0.5,
                                                             0.6,
                                                             0.7,
                                                             0.8,
                                                             0.9,
                                                             1.0],
                                        'repetitions': 500,
                                        'road_repetitions': 20,
                                        'seed_base': 2026091100,
                                        'road_seed_base': 2026191100,
                                        'random_seed_base': 2026291100,
                                        'detour_seed_base': 2026391100,
                                        'main_model_seed': 20260905,
                                        'adaptive_target': 0.1,
                                        'adaptive_targets': [0.05, 0.1, 0.2],
                                        'dense_checkpoints': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                                        'allocation': 'delta/K at each of K declared checkpoints',
                                        'settings': 'One-factor-at-a-time sensitivity around '
                                                    'n=round(.2N), epsilon=.05, delta=.1; no '
                                                    'best-target budget selection.',
                                        'scores': 'Freeze existing source-only HGB/Logistic fits '
                                                  'and splits. Rerun released ITS4SDC inference on '
                                                  'official-precision sequences; refit the '
                                                  'common-road HGB using the fixed historical '
                                                  'subset and split.',
                                        'outcomes': 'Archived public outcomes and existing 30000 '
                                                    'openpilot component executions; new decision '
                                                    'evaluations, no new simulator executions.',
                                        'comparators': ['Random',
                                                        'Risk-order',
                                                        'Diversity',
                                                        'Historical',
                                                        'ITS4SDC',
                                                        'ITS4SDC-order',
                                                        'DETOUR',
                                                        'Full'],
                                        'ablations': ['Uncorrected',
                                                      'Unadjusted adaptive allocation'],
                                        'omitted_statistical_comparisons': ['BT-checked',
                                                                            'CRC',
                                                                            'Grid-HG'],
                                        'recall_references': 'Shortest stable-order prefix '
                                                             'reaching ceil(.95M) failures or '
                                                             'exactly the number found by each FPC '
                                                             'draw; hindsight only. Retain signed '
                                                             'cost differences.',
                                        'budget_handling': 'Record complete FPC feasibility at '
                                                           'each budget; never report complete '
                                                           'recall as a low-budget prefix result.',
                                        'target_met': 'Valid threshold and net skipped/N >= '
                                                      'declared goal, calculated identically for '
                                                      'every fixed and adaptive procedure.',
                                        'uncertainty': 'Paired conditional randomization intervals '
                                                       'within fixed libraries; deterministic '
                                                       'orderings have no repeated-fitting '
                                                       'interval. No pooling as independent '
                                                       'projects.',
                                        'no_failure_case': 'Recall undefined when M=0; miss=0 by '
                                                           'convention; retain execution and '
                                                           'refusal outcomes.',
                                        'accounting': 'Total distinct executions include '
                                                      'calibration and retained tests; report '
                                                      'execution count and recorded-duration '
                                                      'saving separately from measured algorithm '
                                                      'overhead.'},
 'simulator_archive/protocol.json': {'protocol_id': 'revision-20260905-v1',
                                     'created_before_new_results': True,
                                     'purpose': 'Fresh raw-data extraction, model training, '
                                                'calibration and evaluation; no legacy result JSON '
                                                'or score arrays are inputs.',
                                     'main_epsilon': 0.05,
                                     'main_delta': 0.1,
                                     'main_calibration_fraction': 0.2,
                                     'calibration_fractions': [0.05, 0.1, 0.2, 0.3],
                                     'epsilon_sensitivity': [0.01, 0.02, 0.05, 0.1],
                                     'delta_sensitivity': [0.01, 0.05, 0.1],
                                     'repetitions': 500,
                                     'seed_base': 202609050,
                                     'model_seeds': [20260905, 20260906, 20260907],
                                     'model': {'max_iter': 200,
                                               'learning_rate': 0.08,
                                               'max_leaf_nodes': 15,
                                               'l2_regularization': 1.0},
                                     'methods': ['FPC',
                                                 'BT-checked',
                                                 'CRC',
                                                 'Historical',
                                                 'Uncorrected'],
                                     'budget_selection': 'Every grid point is reported. The main '
                                                         'point is 20 percent and does not '
                                                         'maximize utility on target labels.',
                                     'history_cut': 'A 25 percent held-out source partition fixes '
                                                    'the historical threshold before target '
                                                    'scoring.',
                                     'public_data_status': 'Offline replay of published simulator '
                                                           'outcomes; fresh extraction and '
                                                           'statistical experiments, not new '
                                                           'Apollo or BeamNG executions.',
                                     'openpilot': {'candidate_tags': ['v0.9.8',
                                                                      'v0.9.9',
                                                                      'v0.10.0'],
                                                   'pilot_scenarios': 300,
                                                   'formal_scenarios': 3000,
                                                   'pilot_seed': 91201,
                                                   'formal_seed': 91202,
                                                   'time_step_s': 0.05,
                                                   'oracle': 'minimum longitudinal gap below 0.4 m',
                                                   'scope': 'LongitudinalPlanner component with a '
                                                            'shared numerical plant and synthetic '
                                                            'perception messages; no full '
                                                            'perception network'}},
 'simulator_archive/openpilot_protocol.json': {'protocol_id': 'openpilot-formal-20260905-v1',
                                               'stage': 'Frozen after pilot and before formal '
                                                        'scenario outcomes',
                                               'pilot_scenarios': 300,
                                               'pilot_seed': 91201,
                                               'pilot_observed_failures': {'v0.9.8': 2,
                                                                           'v0.9.9': 2,
                                                                           'v0.10.0': 9},
                                               'formal_scenarios': 10000,
                                               'formal_seed': 91202,
                                               'sample_size_reason': 'The pilot failure rate is '
                                                                     'low. Increase the '
                                                                     'independent formal library '
                                                                     'without changing scenario '
                                                                     'distribution or oracle.',
                                               'distribution_unchanged_from_pilot': True,
                                               'oracle': 'A longitudinal gap strictly below 0.4 m '
                                                         'at any simulated step',
                                               'termination': 'First gap violation or 30 simulated '
                                                              'seconds',
                                               'time_step_s': 0.05,
                                               'main_calibration_fraction': 0.2,
                                               'main_epsilon': 0.05,
                                               'main_delta': 0.1,
                                               'candidate_tags': ['v0.9.8', 'v0.9.9', 'v0.10.0'],
                                               'source_target_pairs': [['v0.9.8', 'v0.9.9'],
                                                                       ['v0.9.9', 'v0.10.0'],
                                                                       ['v0.9.8', 'v0.10.0']],
                                               'target_outcomes_do_not_choose_model_or_budget': True,
                                               'replicate_check': 'Re-execute the first 30 formal '
                                                                  'scenarios per version without '
                                                                  'changing inputs.',
                                               'scoring': 'Fit HGB-spec and Logistic-spec on '
                                                          'historical-version results; split '
                                                          'source cases 75/25 before model '
                                                          'training and historical threshold '
                                                          'selection; retain all target scenarios.',
                                               'scope': 'Released longitudinal planning component, '
                                                        'shared numerical vehicle and synthetic '
                                                        'sensor/model messages. No perception '
                                                        'network inference or hardware process '
                                                        'graph.'}}

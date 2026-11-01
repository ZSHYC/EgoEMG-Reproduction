# 原生 EMGFormer 中心测试报告

规模：large；验证最优检查点：best-val_mae=val_mae=0.2554.ckpt。

| 唯一手样本 | 重复消费 | pooled MAE (°) | 有效角度数 |
| ---: | ---: | ---: | ---: |
| 2792 | 0 | 16.573467 | 59576 |

原生按样本数加权批量 MAE：0.2892048657 rad。

保留原生中心标签与生成掩码；总体指标按有效角度数汇总，分别列出左右手、条件、匿名用户、22关节和指组。
采用可用旧版数据与独立训练留出验证；未证明与PDF原表人口等价，未取得全论文完成资格。

```json
{
  "actual_replay_seconds": 42.016198577,
  "actual_test_pass_seconds": 29.241357121,
  "fit_result_sha256": "cf57d3db07807de24d784bacd6a54ace4830c19dee87b362973b17162ba3b92b",
  "format": "original_emgformer_completed_point_report_v1",
  "full_paper_complete": false,
  "independent_test_sha256": "0fe811daf14996641d7c7d538b966d011d1da405cc22b883d401f994d7fe62d6",
  "limitations": [
    "Legacy prefiltered temporal validation: official filtered EMG was processed over whole episodes; disjoint row footprints do not remove upstream filter dependence.",
    "Generated labels are the public release labels; public label-generation code differs from the paper description. No labels are regenerated here.",
    "Validation is a deterministic contiguous temporal holdout within official training runs, not a new unseen-subject benchmark.",
    "Short available training runs with fewer than two full windows contribute training only; validation fraction is approximate."
  ],
  "native_test_metrics": {
    "test_mae": 0.2892048954963684
  },
  "native_test_sha256": "03650e6e86524280f923f482f63cf3d8a359a0fa4afcbadff93e98191f98e088",
  "paper_numerical_equivalence_established": false,
  "population": {
    "counts_describe": "released metadata subject keys, not independently verified natural-person identities",
    "cross_role_user_intersection_counts": {
      "train_test": 36,
      "train_val": 36,
      "val_test": 36
    },
    "episode_records": 41,
    "train_test_user_intersection_by_split": {
      "both": 1,
      "gesture": 36,
      "user": 1
    },
    "unique_subjects_by_role": {
      "test": 40,
      "train": 36,
      "val": 36
    }
  },
  "scientific_eligible": true,
  "scope": "released legacy EgoEMG; native one-consumption point grid; independent train-only validation",
  "sha256": "9f3d4840159b7d34761c2cabee404ae53a0b60baef31621e4348a7ec21f2845b",
  "size": "large",
  "sources_sha256": {
    "egoemg_repro/classic_emg_report.py": "bdddaf770519151c24cd0f8fdbd9f40e4453e04fa1a28e15088ad825af325868",
    "egoemg_repro/classic_emg_train.py": "52b1ce655f3c4e107e3d809e403cd507fbcfd655f7ac9ea7bc0b9a592b18b0d8",
    "egoemg_repro/data.py": "de7a716c6616277f6189c94abb09803f950dd27bc1f9f7193b1a000758a88f0e",
    "egoemg_repro/emgformer_cohort.py": "c076a746abd25eb48fd4eb29135921c5cdeae8c083f9b38d423161a49fe25ea4",
    "egoemg_repro/emgformer_data.py": "c74e4a70760c4c6689d7c7a2f4dc6aa4cde1bbb94fb79d70983ed2cbd27cb4c0",
    "egoemg_repro/emgformer_eval.py": "32f5be097a57793f1c899d0a530873728a6424a556bda9e3ab3825ebb6307480",
    "egoemg_repro/emgformer_literal.py": "da7ba7c1e1035ad520c5ae9cb1ecdf8014f07363d95568d8542ae060941ae701",
    "egoemg_repro/emgformer_runtime.py": "7691be8aacfb6a6796225ee91be9a19643d585a83512134178aa8ea354513bbb",
    "egoemg_repro/emgformer_train.py": "3cc0be18daa1a405594a9c7c4998bd20d0efce26603c6069c823cb4d58e220e6",
    "egoemg_repro/metrics.py": "12d6eb170997ae78ddbf5602d1252d05a778f1357f17b28aa3b96d06a618798e",
    "egoemg_repro/vision_literal_check.py": "9601034285aaea5cb35abed793576ab5ecf901146b55a4c3901f7b242234b227",
    "egoemg_repro/vision_literal_eval.py": "63553900476855dc0cf9ecc2f97f841e33225470488066de39af23465565c581",
    "egoemg_repro/vision_literal_runtime.py": "bb430267f819c1a1c46ed961397a81c3b9619d9a48f321939cc78808bc85e4d0",
    "egoemg_repro/vision_literal_train.py": "b89e20be1e6ef3669f634391ac072c51ecc929c9517fbc244b6a02747c52efd7",
    "egoemg_repro/wilor_native_train.py": "91967a141fad6a00cf19711f8d06749e0409b21f06b8f6325a2808ae8cf011b2"
  },
  "statement": "This single setting is not full-paper completion or proof of PDF population equivalence.",
  "test": {
    "all": {
      "by_component": {
        "fingers": {
          "absolute_error_sum_rad": 15934.70524494446,
          "mae_deg": 16.857299821275742,
          "valid_angle_count": 54160
        },
        "wrist": {
          "absolute_error_sum_rad": 1298.3423020003247,
          "mae_deg": 13.73514295936454,
          "valid_angle_count": 5416
        }
      },
      "by_finger": {
        "index": {
          "absolute_error_sum_rad": 2879.955338571868,
          "mae_deg": 15.233501300437418,
          "valid_angle_count": 10832
        },
        "middle": {
          "absolute_error_sum_rad": 3345.290664383807,
          "mae_deg": 17.694888876819356,
          "valid_angle_count": 10832
        },
        "pinky": {
          "absolute_error_sum_rad": 3737.9283893970933,
          "mae_deg": 19.77174306080011,
          "valid_angle_count": 10832
        },
        "ring": {
          "absolute_error_sum_rad": 3287.9806568379136,
          "mae_deg": 17.39174803890922,
          "valid_angle_count": 10832
        },
        "thumb": {
          "absolute_error_sum_rad": 2683.550195753778,
          "mae_deg": 14.194617829412609,
          "valid_angle_count": 10832
        }
      },
      "by_hand": {
        "left": {
          "absolute_error_sum_rad": 8229.9420349703,
          "mae_deg": 15.841595922902181,
          "valid_angle_count": 29766
        },
        "right": {
          "absolute_error_sum_rad": 9003.105511974485,
          "mae_deg": 17.3042585825933,
          "valid_angle_count": 29810
        }
      },
      "by_hand_split": {
        "left": {
          "both": {
            "absolute_error_sum_rad": 701.0945511887403,
            "mae_deg": 18.258981283060667,
            "valid_angle_count": 2200
          },
          "gesture": {
            "absolute_error_sum_rad": 4100.495030634265,
            "mae_deg": 14.709557924474746,
            "valid_angle_count": 15972
          },
          "user": {
            "absolute_error_sum_rad": 3428.352453147294,
            "mae_deg": 16.942394880857535,
            "valid_angle_count": 11594
          }
        },
        "right": {
          "both": {
            "absolute_error_sum_rad": 761.5305945713189,
            "mae_deg": 19.636583725933743,
            "valid_angle_count": 2222
          },
          "gesture": {
            "absolute_error_sum_rad": 4368.112535774439,
            "mae_deg": 15.691185751602704,
            "valid_angle_count": 15950
          },
          "user": {
            "absolute_error_sum_rad": 3873.4623816287276,
            "mae_deg": 19.06968951452297,
            "valid_angle_count": 11638
          }
        }
      },
      "by_joint": [
        {
          "absolute_error_sum_rad": 563.3924945982726,
          "index": 0,
          "mae_deg": 11.920240823422471,
          "name": "THUMB_CMC_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 770.3699201271957,
          "index": 1,
          "mae_deg": 16.299462735272765,
          "name": "THUMB_CMC_AA",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 693.1761148863006,
          "index": 2,
          "mae_deg": 14.666198612356167,
          "name": "THUMB_MCP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 656.6116661420092,
          "index": 3,
          "mae_deg": 13.892569146599033,
          "name": "THUMB_IP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 363.5354881193489,
          "index": 4,
          "mae_deg": 7.6916725156820425,
          "name": "INDEX_MCP_AA",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 869.8303920289691,
          "index": 5,
          "mae_deg": 18.403844296702278,
          "name": "INDEX_MCP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1175.8197871345328,
          "index": 6,
          "mae_deg": 24.877958371779755,
          "name": "INDEX_PIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 470.7696712890174,
          "index": 7,
          "mae_deg": 9.960530017585592,
          "name": "INDEX_DIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 284.23557342206186,
          "index": 8,
          "mae_deg": 6.0138473945956346,
          "name": "MIDDLE_MCP_AA",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1057.6706103322213,
          "index": 9,
          "mae_deg": 22.378161775133737,
          "name": "MIDDLE_MCP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1369.3584682531073,
          "index": 10,
          "mae_deg": 28.97284374867141,
          "name": "MIDDLE_PIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 634.0260123764165,
          "index": 11,
          "mae_deg": 13.414702588876649,
          "name": "MIDDLE_DIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 267.77418057275645,
          "index": 12,
          "mae_deg": 5.665557758269182,
          "name": "RING_MCP_AA",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1085.6342676583445,
          "index": 13,
          "mae_deg": 22.96981596440144,
          "name": "RING_MCP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1464.7341819246649,
          "index": 14,
          "mae_deg": 30.990800122906432,
          "name": "RING_PIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 469.83802668214776,
          "index": 15,
          "mae_deg": 9.940818310059832,
          "name": "RING_DIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 564.6210952553447,
          "index": 16,
          "mae_deg": 11.946235517793678,
          "name": "PINKY_MCP_AA",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1403.9621375452407,
          "index": 17,
          "mae_deg": 29.70498710395416,
          "name": "PINKY_MCP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1256.8906796434894,
          "index": 18,
          "mae_deg": 26.593253786152715,
          "name": "PINKY_PIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 512.4544769530185,
          "index": 19,
          "mae_deg": 10.842495835299879,
          "name": "PINKY_DIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 784.9700903694611,
          "index": 20,
          "mae_deg": 16.608372681747767,
          "name": "wrist_pitch",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 513.3722116308636,
          "index": 21,
          "mae_deg": 10.861913236981314,
          "name": "wrist_yaw",
          "valid_angle_count": 2708
        }
      ],
      "by_pd_group": {
        "distal": {
          "absolute_error_sum_rad": 2743.6998534426093,
          "mae_deg": 11.610223179684198,
          "valid_angle_count": 13540
        },
        "mid": {
          "absolute_error_sum_rad": 5959.979231842095,
          "mae_deg": 25.220210928373294,
          "valid_angle_count": 13540
        },
        "proximal": {
          "absolute_error_sum_rad": 7231.026159659756,
          "mae_deg": 15.299382588522738,
          "valid_angle_count": 27080
        }
      },
      "by_split": {
        "both": {
          "absolute_error_sum_rad": 1462.6251457600592,
          "mae_deg": 18.951209376245647,
          "valid_angle_count": 4422
        },
        "gesture": {
          "absolute_error_sum_rad": 8468.607566408704,
          "mae_deg": 15.20003357896666,
          "valid_angle_count": 31922
        },
        "user": {
          "absolute_error_sum_rad": 7301.814834776022,
          "mae_deg": 18.00805668124486,
          "valid_angle_count": 23232
        }
      },
      "by_split_user_summary": {
        "both": {
          "aggregation": "unweighted user means; ddof=0",
          "mean_mae_deg": 18.522136010781164,
          "std_mae_deg": 1.8753974169534495,
          "user_count": 5
        },
        "gesture": {
          "aggregation": "unweighted user means; ddof=0",
          "mean_mae_deg": 15.2446593263038,
          "std_mae_deg": 1.9193574808224874,
          "user_count": 36
        },
        "user": {
          "aggregation": "unweighted user means; ddof=0",
          "mean_mae_deg": 17.872788897285982,
          "std_mae_deg": 1.3455249786723804,
          "user_count": 5
        }
      },
      "by_user": {
        "user_000": {
          "absolute_error_sum_rad": 214.88462474707194,
          "mae_deg": 14.727251292176636,
          "valid_angle_count": 836
        },
        "user_001": {
          "absolute_error_sum_rad": 307.1930769183673,
          "mae_deg": 16.00078800278192,
          "valid_angle_count": 1100
        },
        "user_002": {
          "absolute_error_sum_rad": 230.0508087388589,
          "mae_deg": 13.024644678169894,
          "valid_angle_count": 1012
        },
        "user_003": {
          "absolute_error_sum_rad": 200.17727785231546,
          "mae_deg": 13.7192741332001,
          "valid_angle_count": 836
        },
        "user_004": {
          "absolute_error_sum_rad": 291.87396105136577,
          "mae_deg": 17.677744310791788,
          "valid_angle_count": 946
        },
        "user_005": {
          "absolute_error_sum_rad": 260.44922426578705,
          "mae_deg": 13.300036834122874,
          "valid_angle_count": 1122
        },
        "user_006": {
          "absolute_error_sum_rad": 242.9925615011016,
          "mae_deg": 13.757359908187958,
          "valid_angle_count": 1012
        },
        "user_007": {
          "absolute_error_sum_rad": 306.4941848904127,
          "mae_deg": 14.513077057457005,
          "valid_angle_count": 1210
        },
        "user_008": {
          "absolute_error_sum_rad": 337.161594979887,
          "mae_deg": 16.26088923084745,
          "valid_angle_count": 1188
        },
        "user_009": {
          "absolute_error_sum_rad": 224.90781899774447,
          "mae_deg": 14.643487281889767,
          "valid_angle_count": 880
        },
        "user_010": {
          "absolute_error_sum_rad": 307.0266752191819,
          "mae_deg": 14.278679129864434,
          "valid_angle_count": 1232
        },
        "user_011": {
          "absolute_error_sum_rad": 256.9783020262694,
          "mae_deg": 17.612167622659577,
          "valid_angle_count": 836
        },
        "user_012": {
          "absolute_error_sum_rad": 2528.492661689015,
          "mae_deg": 18.92266954605283,
          "valid_angle_count": 7656
        },
        "user_013": {
          "absolute_error_sum_rad": 179.64598353090696,
          "mae_deg": 18.714466659632116,
          "valid_angle_count": 550
        },
        "user_014": {
          "absolute_error_sum_rad": 369.01113398782036,
          "mae_deg": 17.796953342456764,
          "valid_angle_count": 1188
        },
        "user_015": {
          "absolute_error_sum_rad": 360.33308565229527,
          "mae_deg": 17.378421739732758,
          "valid_angle_count": 1188
        },
        "user_016": {
          "absolute_error_sum_rad": 39.8262950014323,
          "mae_deg": 17.2869592214018,
          "valid_angle_count": 132
        },
        "user_017": {
          "absolute_error_sum_rad": 97.24538277368993,
          "mae_deg": 12.663068204696891,
          "valid_angle_count": 440
        },
        "user_018": {
          "absolute_error_sum_rad": 366.0396633395576,
          "mae_deg": 15.375753551133558,
          "valid_angle_count": 1364
        },
        "user_019": {
          "absolute_error_sum_rad": 259.0886240105028,
          "mae_deg": 12.268334442688973,
          "valid_angle_count": 1210
        },
        "user_020": {
          "absolute_error_sum_rad": 356.22613211895805,
          "mae_deg": 14.495918979180367,
          "valid_angle_count": 1408
        },
        "user_021": {
          "absolute_error_sum_rad": 143.82545267112437,
          "mae_deg": 18.728616874122725,
          "valid_angle_count": 440
        },
        "user_022": {
          "absolute_error_sum_rad": 296.8304403982911,
          "mae_deg": 13.804489826162023,
          "valid_angle_count": 1232
        },
        "user_023": {
          "absolute_error_sum_rad": 243.90211169078248,
          "mae_deg": 13.808855350010129,
          "valid_angle_count": 1012
        },
        "user_024": {
          "absolute_error_sum_rad": 1935.8396671921073,
          "mae_deg": 20.00639299136321,
          "valid_angle_count": 5544
        },
        "user_025": {
          "absolute_error_sum_rad": 163.60745475045405,
          "mae_deg": 15.217559503374362,
          "valid_angle_count": 616
        },
        "user_026": {
          "absolute_error_sum_rad": 382.5342795388424,
          "mae_deg": 17.79025952650431,
          "valid_angle_count": 1232
        },
        "user_027": {
          "absolute_error_sum_rad": 386.2812433615327,
          "mae_deg": 16.76688253763763,
          "valid_angle_count": 1320
        },
        "user_028": {
          "absolute_error_sum_rad": 170.3995374609949,
          "mae_deg": 13.05237209558339,
          "valid_angle_count": 748
        },
        "user_029": {
          "absolute_error_sum_rad": 185.57804569974542,
          "mae_deg": 15.103464188751888,
          "valid_angle_count": 704
        },
        "user_030": {
          "absolute_error_sum_rad": 954.2179985893017,
          "mae_deg": 15.532006833689582,
          "valid_angle_count": 3520
        },
        "user_031": {
          "absolute_error_sum_rad": 71.73200108483434,
          "mae_deg": 14.370422790870107,
          "valid_angle_count": 286
        },
        "user_032": {
          "absolute_error_sum_rad": 100.02493667975068,
          "mae_deg": 14.472239186396504,
          "valid_angle_count": 396
        },
        "user_033": {
          "absolute_error_sum_rad": 183.05538637028076,
          "mae_deg": 17.02646275349627,
          "valid_angle_count": 616
        },
        "user_034": {
          "absolute_error_sum_rad": 173.25398649973795,
          "mae_deg": 15.040488197350799,
          "valid_angle_count": 660
        },
        "user_035": {
          "absolute_error_sum_rad": 181.4163547642529,
          "mae_deg": 15.749077973696599,
          "valid_angle_count": 660
        },
        "user_036": {
          "absolute_error_sum_rad": 220.66284906625515,
          "mae_deg": 14.735489448518319,
          "valid_angle_count": 858
        },
        "user_037": {
          "absolute_error_sum_rad": 1774.1828930563206,
          "mae_deg": 16.68086508966811,
          "valid_angle_count": 6094
        },
        "user_038": {
          "absolute_error_sum_rad": 1692.9605229956142,
          "mae_deg": 17.636271427256226,
          "valid_angle_count": 5500
        },
        "user_039": {
          "absolute_error_sum_rad": 236.64331178201974,
          "mae_deg": 17.119524008975,
          "valid_angle_count": 792
        }
      },
      "invalid_angle_count": 1848,
      "mode": "center",
      "overall": {
        "absolute_error_sum_rad": 17233.047546944785,
        "mae_deg": 16.573467379283816,
        "valid_angle_count": 59576
      },
      "user_summary": {
        "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
        "mean_mae_deg": 15.676492144313817,
        "std_mae_deg": 1.9168073009175326,
        "user_count": 40
      }
    },
    "batch_sizes": [
      500,
      500,
      500,
      500,
      500,
      292
    ],
    "consumed_samples": 2792,
    "duplicate_consumptions": 0,
    "format": "original_emgformer_single_consumption_point_statistics_v1",
    "native_batch_weighted_mae_rad": 0.289204865694046,
    "role": "test",
    "unique_samples": 2792
  },
  "training": {
    "completed_epochs": 200,
    "full_training_terminated": true,
    "optimizer_executions": 25000,
    "policy": {
      "batch_size_by_role": {
        "train": 500,
        "val": 500
      },
      "bitwise_training_trajectory_guaranteed": false,
      "early_stopping": {
        "check_on_train_epoch_end": false,
        "mode": "min",
        "monitor": "val_mae",
        "patience": 200,
        "state_key": "EarlyStopping{'monitor': 'val_mae', 'mode': 'min'}"
      },
      "effective_loss_weights": {
        "landmark/fingertip": 0.02,
        "mae": 1.0
      },
      "engineering_train_samples": null,
      "format": "original_egoemg_emgformer_fit_policy_v1",
      "full_paper_complete": false,
      "gradient_clip_val": 1,
      "logger": "original TensorBoardLogger; owned output path/version only",
      "max_epochs": 200,
      "monitor": "val_mae",
      "monitor_mode": "min",
      "native_callback_declarations": [
        {
          "_target_": "pytorch_lightning.callbacks.ModelSummary",
          "max_depth": 2
        },
        {
          "_target_": "pytorch_lightning.callbacks.LearningRateMonitor"
        },
        {
          "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
          "filename": "best-val_mae={val_mae:.4f}",
          "mode": "min",
          "monitor": "val_mae",
          "save_last": true,
          "save_top_k": 3,
          "verbose": true
        },
        {
          "_target_": "pytorch_lightning.callbacks.EarlyStopping",
          "check_on_train_epoch_end": false,
          "mode": "min",
          "monitor": "val_mae",
          "patience": 200,
          "verbose": true
        }
      ],
      "native_global_loader_rng": true,
      "native_optimizer": {
        "_target_": "torch.optim.AdamW",
        "amsgrad": false,
        "betas": [
          0.9,
          0.999
        ],
        "eps": 1e-08,
        "lr": 5e-05,
        "weight_decay": 0.01
      },
      "num_sanity_val_steps": 2,
      "persistent_workers": true,
      "pin_memory": false,
      "precision": "bf16-mixed",
      "prefetch_factor": 1,
      "protocol_sha256": "a419b01bc58d70a9635405c50ecacdd296f4a7f70490157e273b733b6c0fb89c",
      "save_last": true,
      "save_top_k": 3,
      "scheduler_T_max": 200,
      "scheduler_eta_min": 1e-05,
      "scientific_eligible": false,
      "scope": "full",
      "sha256": "89f19e88cbbec0333bf83704a6ae95b148acf755b67f9200fac3a0a101d67305",
      "single_rank": true,
      "size": "large",
      "training_target_shape": [
        22,
        12000
      ],
      "validation_target_shape": [
        22,
        1
      ],
      "workers": 6
    },
    "scope": "full",
    "selected_best": {
      "checkpoint_sha256": "896871c83cb46ac12a755ab4dcee0f89b0995cae8fdcfac6df47ec6820ce1605",
      "file": "best-val_mae=val_mae=0.2554.ckpt",
      "score": 0.255403995513916,
      "validated": true
    }
  },
  "validation_profile_sha256": "48c18a4872af5e8278f1f92e0f6a66f2aa45ccaed8e29a20c74bdde7eb28fd51",
  "vision_subset_reported": false
}
```

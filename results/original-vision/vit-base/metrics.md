# DINOv2 视觉中心评价

完整训练实验

模型：vit_base；训练完成 94 轮，实际更新 52828 次。

验证选择：vit-small-epoch=063-val_mae=0.1074.ckpt；不以最后一轮替代验证最优。

测试消费 8612 手样本，唯一 4306，重复消费 4306。

| 统计人口 | MAE (°) | 有效角度数 | 弧度绝对误差和 |
| --- | ---: | ---: | ---: |
| 全部重复消费 | 6.761261 | 183832 | 21693.3271 |
| 唯一首遍 | 6.761261 | 91916 | 10846.66355 |
| 重复消费中的有图子集 | 6.623680 | 181852 | 21023.0034 |

原生按样本数加权的批量 MAE：0.1179233938 rad。

该指标与按有效角度数汇总的 pooled MAE 分开报告。缺图仍保留原生成标签 mask；不增加腕角过滤。

采用可用旧版 EgoEMG 队列、独立 train 留出验证、单 rank/workers0；未证明与论文原表或缺失的合并训练来源数值等价。
当前报告只验收所列实验，整篇论文复现仍未完成。

## 完整公开证据

以下与 metrics.json 一致，包含逐手、逐条件、匿名逐用户、逐角、指组、用户交集与来源摘要。

```json
{
  "cohort_sha256": "a246c17ebfe5eb9f53a6a4a63e18f661a992a5fdeffb7d3ba1d964f94cf1aa71",
  "fixed_agreement_policy": {
    "native_aggregation": "sample-count weighted original per-batch masked L1, float32 accumulation",
    "native_mae_atol_rad": 2e-06,
    "native_mae_rtol": 2e-06,
    "pooled_aggregation": "float64 absolute-error sum / exact valid-angle count",
    "prediction": "exact typed prediction/target/mask/input replay",
    "scope": "fixed before validation; no test-driven tolerance or checkpoint changes"
  },
  "format": "original_vision_center_completed_report_v1",
  "full_paper_complete": false,
  "independent_test_sha256": "cac9805a0be9b4e37bd3d5cc9d2ae524f1c57d6faa52fd7c211bc1ab8f69a5a6",
  "native_metrics": {
    "test_derivatives/acc": 0.0,
    "test_derivatives/jerk": 0.0,
    "test_derivatives/vel": 0.0,
    "test_landmark/all": 9.47365951538086,
    "test_landmark/fingertip": 15.133139610290527,
    "test_loss": 0.42058616876602173,
    "test_mae": 0.11792339384555817,
    "test_mae_per_finger/index": 0.1042497381567955,
    "test_mae_per_finger/middle": 0.11022652685642242,
    "test_mae_per_finger/pinky": 0.15356087684631348,
    "test_mae_per_finger/ring": 0.11131974309682846,
    "test_mae_per_finger/thumb": 0.10799247771501541,
    "test_mae_per_pd/distal": 0.09252955764532089,
    "test_mae_per_pd/mid": 0.14586855471134186,
    "test_mae_per_pd/proximal": 0.11574071645736694,
    "test_mae_wrist/deviation": 0.1288788914680481,
    "test_mae_wrist/flexion": 0.11603822559118271
  },
  "native_test_sha256": "596c2002bf98373e25305a8ce345e2201a76af56e4e83efddd13d134d5f47415",
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
      "test": 41,
      "train": 36,
      "val": 36
    }
  },
  "prediction_sha256": "05fb99fae4493cb4a11c25f1930140eb958ac822f85510f1476fc540dbb182bc",
  "scientific_eligible": true,
  "scope": "available legacy cohort; single rank/workers0; original vision semantics; independent train-only validation",
  "sha256": "3f64e962d15cb3f2183803a7a9cfb66455ebe814a3ff4563863a0f03ad1202cc",
  "sources_sha256": {
    "egoemg_repro/classic_emg_report.py": "bdddaf770519151c24cd0f8fdbd9f40e4453e04fa1a28e15088ad825af325868",
    "egoemg_repro/metrics.py": "12d6eb170997ae78ddbf5602d1252d05a778f1357f17b28aa3b96d06a618798e",
    "egoemg_repro/vision_literal_eval.py": "329f6e74aa2424082abcdd9d983da879a54d09d1ad341d1d40d1f2e9941c18d8",
    "egoemg_repro/vision_literal_train.py": "e44dc947111bec3aa858fb5fc36493650cda24f2093915bcd133a2d090d070da"
  },
  "test": {
    "batch_sizes": [
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      360,
      332
    ],
    "consumed": {
      "all": {
        "by_component": {
          "fingers": {
            "absolute_error_sum_rad": 19641.629725060564,
            "mae_deg": 6.73397849452294,
            "valid_angle_count": 167120
          },
          "wrist": {
            "absolute_error_sum_rad": 2051.6973772273523,
            "mae_deg": 7.034083326543063,
            "valid_angle_count": 16712
          }
        },
        "by_finger": {
          "index": {
            "absolute_error_sum_rad": 3485.680155399663,
            "mae_deg": 5.975190331405745,
            "valid_angle_count": 33424
          },
          "middle": {
            "absolute_error_sum_rad": 3683.125212753046,
            "mae_deg": 6.313652767740933,
            "valid_angle_count": 33424
          },
          "pinky": {
            "absolute_error_sum_rad": 5134.048136810568,
            "mae_deg": 8.800840415756625,
            "valid_angle_count": 33424
          },
          "ring": {
            "absolute_error_sum_rad": 3726.125711090204,
            "mae_deg": 6.387364683480475,
            "valid_angle_count": 33424
          },
          "thumb": {
            "absolute_error_sum_rad": 3612.6505090070823,
            "mae_deg": 6.192844274230924,
            "valid_angle_count": 33424
          }
        },
        "by_hand": {
          "left": {
            "absolute_error_sum_rad": 11201.271354862789,
            "mae_deg": 6.958986530777491,
            "valid_angle_count": 92224
          },
          "right": {
            "absolute_error_sum_rad": 10492.055747425127,
            "mae_deg": 6.562205405023996,
            "valid_angle_count": 91608
          }
        },
        "by_hand_split": {
          "left": {
            "both": {
              "absolute_error_sum_rad": 918.7518040057039,
              "mae_deg": 7.669085196613845,
              "valid_angle_count": 6864
            },
            "gesture": {
              "absolute_error_sum_rad": 5415.222579891553,
              "mae_deg": 6.284828207578416,
              "valid_angle_count": 49368
            },
            "user": {
              "absolute_error_sum_rad": 4867.296970965532,
              "mae_deg": 7.7482655610450815,
              "valid_angle_count": 35992
            }
          },
          "right": {
            "both": {
              "absolute_error_sum_rad": 854.8299242850808,
              "mae_deg": 7.090061792562659,
              "valid_angle_count": 6908
            },
            "gesture": {
              "absolute_error_sum_rad": 5201.214451793479,
              "mae_deg": 6.101712457621141,
              "valid_angle_count": 48840
            },
            "user": {
              "absolute_error_sum_rad": 4436.011371346567,
              "mae_deg": 7.087694630513075,
              "valid_angle_count": 35860
            }
          }
        },
        "by_joint": [
          {
            "absolute_error_sum_rad": 686.3475801796194,
            "index": 0,
            "mae_deg": 4.706177551856039,
            "name": "THUMB_CMC_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 867.4242302431885,
            "index": 1,
            "mae_deg": 5.947791699415855,
            "name": "THUMB_CMC_AA",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 995.3331538755447,
            "index": 2,
            "mae_deg": 6.824843097955247,
            "name": "THUMB_MCP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1063.5455447087297,
            "index": 3,
            "mae_deg": 7.292564747696557,
            "name": "THUMB_IP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 735.7570362837287,
            "index": 4,
            "mae_deg": 5.04497043155953,
            "name": "INDEX_MCP_AA",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 876.2229507951415,
            "index": 5,
            "mae_deg": 6.008123144215032,
            "name": "INDEX_MCP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1204.2321907196892,
            "index": 6,
            "mae_deg": 8.25723098157389,
            "name": "INDEX_PIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 669.4679776011035,
            "index": 7,
            "mae_deg": 4.590436768274529,
            "name": "INDEX_DIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 717.9604842642439,
            "index": 8,
            "mae_deg": 4.92294226968764,
            "name": "MIDDLE_MCP_AA",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1059.244697512855,
            "index": 9,
            "mae_deg": 7.263074514013658,
            "name": "MIDDLE_MCP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1128.159559115069,
            "index": 10,
            "mae_deg": 7.735612895480273,
            "name": "MIDDLE_PIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 777.7604718608782,
            "index": 11,
            "mae_deg": 5.332981391782161,
            "name": "MIDDLE_DIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 640.7152133438317,
            "index": 12,
            "mae_deg": 4.3932835799935,
            "name": "RING_MCP_AA",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1175.714141659846,
            "index": 13,
            "mae_deg": 8.06168719853463,
            "name": "RING_MCP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1319.9209884293377,
            "index": 14,
            "mae_deg": 9.050490896091075,
            "name": "RING_PIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 589.7753676571883,
            "index": 15,
            "mae_deg": 4.043997059302695,
            "name": "RING_DIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1139.4054828538792,
            "index": 16,
            "mae_deg": 7.8127244281465975,
            "name": "PINKY_MCP_AA",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1775.5198532563227,
            "index": 17,
            "mae_deg": 12.174460750750903,
            "name": "PINKY_MCP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1453.2590835792944,
            "index": 18,
            "mae_deg": 9.964769270960188,
            "name": "PINKY_PIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 765.8637171210721,
            "index": 19,
            "mae_deg": 5.251407213168814,
            "name": "PINKY_DIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 971.0317066772113,
            "index": 20,
            "mae_deg": 6.6582118915736626,
            "name": "wrist_pitch",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1080.665670550141,
            "index": 21,
            "mae_deg": 7.409954761512463,
            "name": "wrist_yaw",
            "valid_angle_count": 8356
          }
        ],
        "by_pd_group": {
          "distal": {
            "absolute_error_sum_rad": 3866.413078948972,
            "mae_deg": 5.3022774360449505,
            "valid_angle_count": 41780
          },
          "mid": {
            "absolute_error_sum_rad": 6100.904975718935,
            "mae_deg": 8.366589428412134,
            "valid_angle_count": 41780
          },
          "proximal": {
            "absolute_error_sum_rad": 9674.311670392657,
            "mae_deg": 6.633523556817338,
            "valid_angle_count": 83560
          }
        },
        "by_split": {
          "both": {
            "absolute_error_sum_rad": 1773.5817282907847,
            "mae_deg": 7.3786485370737935,
            "valid_angle_count": 13772
          },
          "gesture": {
            "absolute_error_sum_rad": 10616.437031685033,
            "mae_deg": 6.19376257923945,
            "valid_angle_count": 98208
          },
          "user": {
            "absolute_error_sum_rad": 9303.308342312099,
            "mae_deg": 7.41858686638275,
            "valid_angle_count": 71852
          }
        },
        "by_split_user_summary": {
          "both": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 7.365534469663622,
            "std_mae_deg": 0.3449883180528243,
            "user_count": 5
          },
          "gesture": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 6.252778288946048,
            "std_mae_deg": 1.384176640736676,
            "user_count": 36
          },
          "user": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 8.931965562015474,
            "std_mae_deg": 3.216250413995589,
            "user_count": 6
          }
        },
        "by_user": {
          "user_000": {
            "absolute_error_sum_rad": 255.17233610991389,
            "mae_deg": 5.933562462496656,
            "valid_angle_count": 2464
          },
          "user_001": {
            "absolute_error_sum_rad": 385.6060036830604,
            "mae_deg": 6.437528136930489,
            "valid_angle_count": 3432
          },
          "user_002": {
            "absolute_error_sum_rad": 544.451907799812,
            "mae_deg": 10.742009801919785,
            "valid_angle_count": 2904
          },
          "user_003": {
            "absolute_error_sum_rad": 24.683485438348725,
            "mae_deg": 16.07113112829553,
            "valid_angle_count": 88
          },
          "user_004": {
            "absolute_error_sum_rad": 373.5697792262072,
            "mae_deg": 7.721490513454403,
            "valid_angle_count": 2772
          },
          "user_005": {
            "absolute_error_sum_rad": 330.5095430354995,
            "mae_deg": 7.055440351986428,
            "valid_angle_count": 2684
          },
          "user_006": {
            "absolute_error_sum_rad": 294.1125668056193,
            "mae_deg": 5.17549409696605,
            "valid_angle_count": 3256
          },
          "user_007": {
            "absolute_error_sum_rad": 351.38484296086244,
            "mae_deg": 6.355072123271664,
            "valid_angle_count": 3168
          },
          "user_008": {
            "absolute_error_sum_rad": 401.79766675201245,
            "mae_deg": 6.228709557114241,
            "valid_angle_count": 3696
          },
          "user_009": {
            "absolute_error_sum_rad": 418.39924396097194,
            "mae_deg": 6.409762254125185,
            "valid_angle_count": 3740
          },
          "user_010": {
            "absolute_error_sum_rad": 261.87317163011176,
            "mae_deg": 5.500083395201777,
            "valid_angle_count": 2728
          },
          "user_011": {
            "absolute_error_sum_rad": 497.67967631842475,
            "mae_deg": 7.364396953120518,
            "valid_angle_count": 3872
          },
          "user_012": {
            "absolute_error_sum_rad": 264.67376697713917,
            "mae_deg": 5.942276565681463,
            "valid_angle_count": 2552
          },
          "user_013": {
            "absolute_error_sum_rad": 3030.7663853426347,
            "mae_deg": 7.281538182248116,
            "valid_angle_count": 23848
          },
          "user_014": {
            "absolute_error_sum_rad": 233.30797471664846,
            "mae_deg": 6.329338199814808,
            "valid_angle_count": 2112
          },
          "user_015": {
            "absolute_error_sum_rad": 390.0155740912887,
            "mae_deg": 6.4287244936141255,
            "valid_angle_count": 3476
          },
          "user_016": {
            "absolute_error_sum_rad": 300.9758553788997,
            "mae_deg": 4.899047229701929,
            "valid_angle_count": 3520
          },
          "user_017": {
            "absolute_error_sum_rad": 31.89077869988978,
            "mae_deg": 5.190929047981213,
            "valid_angle_count": 352
          },
          "user_018": {
            "absolute_error_sum_rad": 147.88366109132767,
            "mae_deg": 6.419022454148561,
            "valid_angle_count": 1320
          },
          "user_019": {
            "absolute_error_sum_rad": 381.631496517919,
            "mae_deg": 5.0709355472491495,
            "valid_angle_count": 4312
          },
          "user_020": {
            "absolute_error_sum_rad": 374.3847111359937,
            "mae_deg": 6.018704787406541,
            "valid_angle_count": 3564
          },
          "user_021": {
            "absolute_error_sum_rad": 394.20708175509935,
            "mae_deg": 5.2380338679813985,
            "valid_angle_count": 4312
          },
          "user_022": {
            "absolute_error_sum_rad": 203.20726205362007,
            "mae_deg": 8.820392789455573,
            "valid_angle_count": 1320
          },
          "user_023": {
            "absolute_error_sum_rad": 334.93799158593174,
            "mae_deg": 5.192243862679158,
            "valid_angle_count": 3696
          },
          "user_024": {
            "absolute_error_sum_rad": 292.2047589054564,
            "mae_deg": 5.284753610770331,
            "valid_angle_count": 3168
          },
          "user_025": {
            "absolute_error_sum_rad": 2324.502141200581,
            "mae_deg": 7.72171626622891,
            "valid_angle_count": 17248
          },
          "user_026": {
            "absolute_error_sum_rad": 238.47682234435342,
            "mae_deg": 7.393785406938624,
            "valid_angle_count": 1848
          },
          "user_027": {
            "absolute_error_sum_rad": 480.57369521900546,
            "mae_deg": 6.953243555822061,
            "valid_angle_count": 3960
          },
          "user_028": {
            "absolute_error_sum_rad": 327.5868575607892,
            "mae_deg": 4.847454638994443,
            "valid_angle_count": 3872
          },
          "user_029": {
            "absolute_error_sum_rad": 235.77203425415792,
            "mae_deg": 6.140337495444148,
            "valid_angle_count": 2200
          },
          "user_030": {
            "absolute_error_sum_rad": 170.5225441204384,
            "mae_deg": 4.441010040879314,
            "valid_angle_count": 2200
          },
          "user_031": {
            "absolute_error_sum_rad": 1421.8352181284463,
            "mae_deg": 7.3471462086781,
            "valid_angle_count": 11088
          },
          "user_032": {
            "absolute_error_sum_rad": 119.08233677002136,
            "mae_deg": 7.384107479954285,
            "valid_angle_count": 924
          },
          "user_033": {
            "absolute_error_sum_rad": 123.71606392692775,
            "mae_deg": 5.370006303775492,
            "valid_angle_count": 1320
          },
          "user_034": {
            "absolute_error_sum_rad": 187.82465298441093,
            "mae_deg": 5.558656975473145,
            "valid_angle_count": 1936
          },
          "user_035": {
            "absolute_error_sum_rad": 186.52883806929458,
            "mae_deg": 5.280294060696511,
            "valid_angle_count": 2024
          },
          "user_036": {
            "absolute_error_sum_rad": 212.50622517010197,
            "mae_deg": 6.289106313276647,
            "valid_angle_count": 1936
          },
          "user_037": {
            "absolute_error_sum_rad": 234.53400692838477,
            "mae_deg": 4.925882972613071,
            "valid_angle_count": 2728
          },
          "user_038": {
            "absolute_error_sum_rad": 2252.090163817891,
            "mae_deg": 6.884083518442773,
            "valid_angle_count": 18744
          },
          "user_039": {
            "absolute_error_sum_rad": 2212.864023647147,
            "mae_deg": 7.50401096066046,
            "valid_angle_count": 16896
          },
          "user_040": {
            "absolute_error_sum_rad": 445.5839561732719,
            "mae_deg": 10.00394988537254,
            "valid_angle_count": 2552
          }
        },
        "invalid_angle_count": 5632,
        "mode": "center",
        "overall": {
          "absolute_error_sum_rad": 21693.32710228792,
          "mae_deg": 6.761260751979317,
          "valid_angle_count": 183832
        },
        "user_summary": {
          "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
          "mean_mae_deg": 6.662327158460137,
          "std_mae_deg": 1.991625044456936,
          "user_count": 41
        }
      },
      "definition": "all requested labeled centers; vision_valid is the identical crop-present subset",
      "missing_by_hand": {
        "left": 160,
        "right": 186
      },
      "missing_by_split": {
        "both": 18,
        "gesture": 248,
        "user": 80
      },
      "samples": {
        "all": 8612,
        "missing_crop": 346,
        "vision_valid": 8266
      },
      "vision_valid": {
        "by_component": {
          "fingers": {
            "absolute_error_sum_rad": 19018.88689093352,
            "mae_deg": 6.591470783251746,
            "valid_angle_count": 165320
          },
          "wrist": {
            "absolute_error_sum_rad": 2004.11650412354,
            "mae_deg": 6.945766836365333,
            "valid_angle_count": 16532
          }
        },
        "by_hand": {
          "left": {
            "absolute_error_sum_rad": 10853.028503932825,
            "mae_deg": 6.817444285955059,
            "valid_angle_count": 91212
          },
          "right": {
            "absolute_error_sum_rad": 10169.974891124237,
            "mae_deg": 6.42869195736361,
            "valid_angle_count": 90640
          }
        },
        "by_hand_split": {
          "left": {
            "both": {
              "absolute_error_sum_rad": 918.7518040057039,
              "mae_deg": 7.669085196613845,
              "valid_angle_count": 6864
            },
            "gesture": {
              "absolute_error_sum_rad": 5149.129544586169,
              "mae_deg": 6.0624566634659525,
              "valid_angle_count": 48664
            },
            "user": {
              "absolute_error_sum_rad": 4785.147155340952,
              "mae_deg": 7.683240005326428,
              "valid_angle_count": 35684
            }
          },
          "right": {
            "both": {
              "absolute_error_sum_rad": 854.8299242850808,
              "mae_deg": 7.090061792562659,
              "valid_angle_count": 6908
            },
            "gesture": {
              "absolute_error_sum_rad": 4947.14723668051,
              "mae_deg": 5.888537836371967,
              "valid_angle_count": 48136
            },
            "user": {
              "absolute_error_sum_rad": 4367.997730158646,
              "mae_deg": 7.030785337139393,
              "valid_angle_count": 35596
            }
          }
        },
        "by_split": {
          "both": {
            "absolute_error_sum_rad": 1773.5817282907847,
            "mae_deg": 7.3786485370737935,
            "valid_angle_count": 13772
          },
          "gesture": {
            "absolute_error_sum_rad": 10096.276781266679,
            "mae_deg": 5.975971573992853,
            "valid_angle_count": 96800
          },
          "user": {
            "absolute_error_sum_rad": 9153.144885499598,
            "mae_deg": 7.357415421028088,
            "valid_angle_count": 71280
          }
        },
        "by_user": {
          "user_000": {
            "absolute_error_sum_rad": 255.17233610991389,
            "mae_deg": 5.933562462496656,
            "valid_angle_count": 2464
          },
          "user_001": {
            "absolute_error_sum_rad": 336.0942252315581,
            "mae_deg": 5.758606646078822,
            "valid_angle_count": 3344
          },
          "user_002": {
            "absolute_error_sum_rad": 544.451907799812,
            "mae_deg": 10.742009801919785,
            "valid_angle_count": 2904
          },
          "user_003": {
            "absolute_error_sum_rad": 14.194796386407688,
            "mae_deg": 18.484134636107104,
            "valid_angle_count": 44
          },
          "user_004": {
            "absolute_error_sum_rad": 191.06300271640066,
            "mae_deg": 5.07750634357437,
            "valid_angle_count": 2156
          },
          "user_005": {
            "absolute_error_sum_rad": 330.5095430354995,
            "mae_deg": 7.055440351986428,
            "valid_angle_count": 2684
          },
          "user_006": {
            "absolute_error_sum_rad": 269.7496484417352,
            "mae_deg": 4.878635221227629,
            "valid_angle_count": 3168
          },
          "user_007": {
            "absolute_error_sum_rad": 351.38484296086244,
            "mae_deg": 6.355072123271664,
            "valid_angle_count": 3168
          },
          "user_008": {
            "absolute_error_sum_rad": 401.79766675201245,
            "mae_deg": 6.228709557114241,
            "valid_angle_count": 3696
          },
          "user_009": {
            "absolute_error_sum_rad": 418.39924396097194,
            "mae_deg": 6.409762254125185,
            "valid_angle_count": 3740
          },
          "user_010": {
            "absolute_error_sum_rad": 261.87317163011176,
            "mae_deg": 5.500083395201777,
            "valid_angle_count": 2728
          },
          "user_011": {
            "absolute_error_sum_rad": 497.67967631842475,
            "mae_deg": 7.364396953120518,
            "valid_angle_count": 3872
          },
          "user_012": {
            "absolute_error_sum_rad": 264.67376697713917,
            "mae_deg": 5.942276565681463,
            "valid_angle_count": 2552
          },
          "user_013": {
            "absolute_error_sum_rad": 3030.7663853426347,
            "mae_deg": 7.281538182248116,
            "valid_angle_count": 23848
          },
          "user_014": {
            "absolute_error_sum_rad": 233.30797471664846,
            "mae_deg": 6.329338199814808,
            "valid_angle_count": 2112
          },
          "user_015": {
            "absolute_error_sum_rad": 390.0155740912887,
            "mae_deg": 6.4287244936141255,
            "valid_angle_count": 3476
          },
          "user_016": {
            "absolute_error_sum_rad": 300.9758553788997,
            "mae_deg": 4.899047229701929,
            "valid_angle_count": 3520
          },
          "user_017": {
            "absolute_error_sum_rad": 31.89077869988978,
            "mae_deg": 5.190929047981213,
            "valid_angle_count": 352
          },
          "user_018": {
            "absolute_error_sum_rad": 147.88366109132767,
            "mae_deg": 6.419022454148561,
            "valid_angle_count": 1320
          },
          "user_019": {
            "absolute_error_sum_rad": 381.631496517919,
            "mae_deg": 5.0709355472491495,
            "valid_angle_count": 4312
          },
          "user_020": {
            "absolute_error_sum_rad": 374.3847111359937,
            "mae_deg": 6.018704787406541,
            "valid_angle_count": 3564
          },
          "user_021": {
            "absolute_error_sum_rad": 394.20708175509935,
            "mae_deg": 5.2380338679813985,
            "valid_angle_count": 4312
          },
          "user_022": {
            "absolute_error_sum_rad": 203.20726205362007,
            "mae_deg": 8.820392789455573,
            "valid_angle_count": 1320
          },
          "user_023": {
            "absolute_error_sum_rad": 334.93799158593174,
            "mae_deg": 5.192243862679158,
            "valid_angle_count": 3696
          },
          "user_024": {
            "absolute_error_sum_rad": 292.2047589054564,
            "mae_deg": 5.284753610770331,
            "valid_angle_count": 3168
          },
          "user_025": {
            "absolute_error_sum_rad": 2280.4918441852715,
            "mae_deg": 7.653617495654979,
            "valid_angle_count": 17072
          },
          "user_026": {
            "absolute_error_sum_rad": 213.887602241477,
            "mae_deg": 6.962986874209954,
            "valid_angle_count": 1760
          },
          "user_027": {
            "absolute_error_sum_rad": 480.57369521900546,
            "mae_deg": 6.953243555822061,
            "valid_angle_count": 3960
          },
          "user_028": {
            "absolute_error_sum_rad": 327.5868575607892,
            "mae_deg": 4.847454638994443,
            "valid_angle_count": 3872
          },
          "user_029": {
            "absolute_error_sum_rad": 235.77203425415792,
            "mae_deg": 6.140337495444148,
            "valid_angle_count": 2200
          },
          "user_030": {
            "absolute_error_sum_rad": 170.5225441204384,
            "mae_deg": 4.441010040879314,
            "valid_angle_count": 2200
          },
          "user_031": {
            "absolute_error_sum_rad": 1326.1707473831957,
            "mae_deg": 7.077495039005885,
            "valid_angle_count": 10736
          },
          "user_032": {
            "absolute_error_sum_rad": 92.79577381873969,
            "mae_deg": 6.359816024478907,
            "valid_angle_count": 836
          },
          "user_033": {
            "absolute_error_sum_rad": 123.71606392692775,
            "mae_deg": 5.370006303775492,
            "valid_angle_count": 1320
          },
          "user_034": {
            "absolute_error_sum_rad": 187.82465298441093,
            "mae_deg": 5.558656975473145,
            "valid_angle_count": 1936
          },
          "user_035": {
            "absolute_error_sum_rad": 186.52883806929458,
            "mae_deg": 5.280294060696511,
            "valid_angle_count": 2024
          },
          "user_036": {
            "absolute_error_sum_rad": 212.50622517010197,
            "mae_deg": 6.289106313276647,
            "valid_angle_count": 1936
          },
          "user_037": {
            "absolute_error_sum_rad": 234.53400692838477,
            "mae_deg": 4.925882972613071,
            "valid_angle_count": 2728
          },
          "user_038": {
            "absolute_error_sum_rad": 2252.090163817891,
            "mae_deg": 6.884083518442773,
            "valid_angle_count": 18744
          },
          "user_039": {
            "absolute_error_sum_rad": 2212.864023647147,
            "mae_deg": 7.50401096066046,
            "valid_angle_count": 16896
          },
          "user_040": {
            "absolute_error_sum_rad": 232.68096213426907,
            "mae_deg": 6.312328173928474,
            "valid_angle_count": 2112
          }
        },
        "invalid_angle_count": 0,
        "mode": "center",
        "overall": {
          "absolute_error_sum_rad": 21023.003395057065,
          "mae_deg": 6.623679515352983,
          "valid_angle_count": 181852
        },
        "user_summary": {
          "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
          "mean_mae_deg": 6.4991266055686046,
          "std_mae_deg": 2.2280884774754397,
          "user_count": 41
        }
      }
    },
    "consumed_samples": 8612,
    "duplicate_consumptions": 4306,
    "format": "original_vision_center_statistics_v1",
    "native_batch_weighted_mae_rad": 0.11792339384555817,
    "role": "test",
    "unique_first_repeat": {
      "all": {
        "by_component": {
          "fingers": {
            "absolute_error_sum_rad": 9820.814862530282,
            "mae_deg": 6.73397849452294,
            "valid_angle_count": 83560
          },
          "wrist": {
            "absolute_error_sum_rad": 1025.8486886136761,
            "mae_deg": 7.034083326543063,
            "valid_angle_count": 8356
          }
        },
        "by_finger": {
          "index": {
            "absolute_error_sum_rad": 1742.8400776998315,
            "mae_deg": 5.975190331405745,
            "valid_angle_count": 16712
          },
          "middle": {
            "absolute_error_sum_rad": 1841.562606376523,
            "mae_deg": 6.313652767740933,
            "valid_angle_count": 16712
          },
          "pinky": {
            "absolute_error_sum_rad": 2567.024068405284,
            "mae_deg": 8.800840415756625,
            "valid_angle_count": 16712
          },
          "ring": {
            "absolute_error_sum_rad": 1863.062855545102,
            "mae_deg": 6.387364683480475,
            "valid_angle_count": 16712
          },
          "thumb": {
            "absolute_error_sum_rad": 1806.3252545035411,
            "mae_deg": 6.192844274230924,
            "valid_angle_count": 16712
          }
        },
        "by_hand": {
          "left": {
            "absolute_error_sum_rad": 5600.6356774313945,
            "mae_deg": 6.958986530777491,
            "valid_angle_count": 46112
          },
          "right": {
            "absolute_error_sum_rad": 5246.027873712564,
            "mae_deg": 6.562205405023996,
            "valid_angle_count": 45804
          }
        },
        "by_hand_split": {
          "left": {
            "both": {
              "absolute_error_sum_rad": 459.37590200285194,
              "mae_deg": 7.669085196613845,
              "valid_angle_count": 3432
            },
            "gesture": {
              "absolute_error_sum_rad": 2707.6112899457767,
              "mae_deg": 6.284828207578416,
              "valid_angle_count": 24684
            },
            "user": {
              "absolute_error_sum_rad": 2433.648485482766,
              "mae_deg": 7.7482655610450815,
              "valid_angle_count": 17996
            }
          },
          "right": {
            "both": {
              "absolute_error_sum_rad": 427.4149621425404,
              "mae_deg": 7.090061792562659,
              "valid_angle_count": 3454
            },
            "gesture": {
              "absolute_error_sum_rad": 2600.6072258967397,
              "mae_deg": 6.101712457621141,
              "valid_angle_count": 24420
            },
            "user": {
              "absolute_error_sum_rad": 2218.0056856732836,
              "mae_deg": 7.087694630513075,
              "valid_angle_count": 17930
            }
          }
        },
        "by_joint": [
          {
            "absolute_error_sum_rad": 343.1737900898097,
            "index": 0,
            "mae_deg": 4.706177551856039,
            "name": "THUMB_CMC_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 433.71211512159425,
            "index": 1,
            "mae_deg": 5.947791699415855,
            "name": "THUMB_CMC_AA",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 497.66657693777233,
            "index": 2,
            "mae_deg": 6.824843097955247,
            "name": "THUMB_MCP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 531.7727723543649,
            "index": 3,
            "mae_deg": 7.292564747696557,
            "name": "THUMB_IP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 367.87851814186433,
            "index": 4,
            "mae_deg": 5.04497043155953,
            "name": "INDEX_MCP_AA",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 438.11147539757076,
            "index": 5,
            "mae_deg": 6.008123144215032,
            "name": "INDEX_MCP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 602.1160953598446,
            "index": 6,
            "mae_deg": 8.25723098157389,
            "name": "INDEX_PIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 334.73398880055174,
            "index": 7,
            "mae_deg": 4.590436768274529,
            "name": "INDEX_DIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 358.98024213212193,
            "index": 8,
            "mae_deg": 4.92294226968764,
            "name": "MIDDLE_MCP_AA",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 529.6223487564275,
            "index": 9,
            "mae_deg": 7.263074514013658,
            "name": "MIDDLE_MCP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 564.0797795575345,
            "index": 10,
            "mae_deg": 7.735612895480273,
            "name": "MIDDLE_PIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 388.8802359304391,
            "index": 11,
            "mae_deg": 5.332981391782161,
            "name": "MIDDLE_DIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 320.35760667191585,
            "index": 12,
            "mae_deg": 4.3932835799935,
            "name": "RING_MCP_AA",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 587.857070829923,
            "index": 13,
            "mae_deg": 8.06168719853463,
            "name": "RING_MCP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 659.9604942146689,
            "index": 14,
            "mae_deg": 9.050490896091075,
            "name": "RING_PIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 294.88768382859416,
            "index": 15,
            "mae_deg": 4.043997059302695,
            "name": "RING_DIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 569.7027414269396,
            "index": 16,
            "mae_deg": 7.8127244281465975,
            "name": "PINKY_MCP_AA",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 887.7599266281613,
            "index": 17,
            "mae_deg": 12.174460750750903,
            "name": "PINKY_MCP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 726.6295417896472,
            "index": 18,
            "mae_deg": 9.964769270960188,
            "name": "PINKY_PIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 382.93185856053606,
            "index": 19,
            "mae_deg": 5.251407213168814,
            "name": "PINKY_DIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 485.51585333860567,
            "index": 20,
            "mae_deg": 6.6582118915736626,
            "name": "wrist_pitch",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 540.3328352750705,
            "index": 21,
            "mae_deg": 7.409954761512463,
            "name": "wrist_yaw",
            "valid_angle_count": 4178
          }
        ],
        "by_pd_group": {
          "distal": {
            "absolute_error_sum_rad": 1933.206539474486,
            "mae_deg": 5.3022774360449505,
            "valid_angle_count": 20890
          },
          "mid": {
            "absolute_error_sum_rad": 3050.4524878594675,
            "mae_deg": 8.366589428412134,
            "valid_angle_count": 20890
          },
          "proximal": {
            "absolute_error_sum_rad": 4837.155835196329,
            "mae_deg": 6.633523556817338,
            "valid_angle_count": 41780
          }
        },
        "by_split": {
          "both": {
            "absolute_error_sum_rad": 886.7908641453923,
            "mae_deg": 7.3786485370737935,
            "valid_angle_count": 6886
          },
          "gesture": {
            "absolute_error_sum_rad": 5308.218515842516,
            "mae_deg": 6.19376257923945,
            "valid_angle_count": 49104
          },
          "user": {
            "absolute_error_sum_rad": 4651.654171156049,
            "mae_deg": 7.41858686638275,
            "valid_angle_count": 35926
          }
        },
        "by_split_user_summary": {
          "both": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 7.365534469663622,
            "std_mae_deg": 0.3449883180528243,
            "user_count": 5
          },
          "gesture": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 6.252778288946048,
            "std_mae_deg": 1.384176640736676,
            "user_count": 36
          },
          "user": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 8.931965562015474,
            "std_mae_deg": 3.216250413995589,
            "user_count": 6
          }
        },
        "by_user": {
          "user_000": {
            "absolute_error_sum_rad": 127.58616805495694,
            "mae_deg": 5.933562462496656,
            "valid_angle_count": 1232
          },
          "user_001": {
            "absolute_error_sum_rad": 192.8030018415302,
            "mae_deg": 6.437528136930489,
            "valid_angle_count": 1716
          },
          "user_002": {
            "absolute_error_sum_rad": 272.225953899906,
            "mae_deg": 10.742009801919785,
            "valid_angle_count": 1452
          },
          "user_003": {
            "absolute_error_sum_rad": 12.341742719174363,
            "mae_deg": 16.07113112829553,
            "valid_angle_count": 44
          },
          "user_004": {
            "absolute_error_sum_rad": 186.7848896131036,
            "mae_deg": 7.721490513454403,
            "valid_angle_count": 1386
          },
          "user_005": {
            "absolute_error_sum_rad": 165.25477151774976,
            "mae_deg": 7.055440351986428,
            "valid_angle_count": 1342
          },
          "user_006": {
            "absolute_error_sum_rad": 147.05628340280964,
            "mae_deg": 5.17549409696605,
            "valid_angle_count": 1628
          },
          "user_007": {
            "absolute_error_sum_rad": 175.69242148043122,
            "mae_deg": 6.355072123271664,
            "valid_angle_count": 1584
          },
          "user_008": {
            "absolute_error_sum_rad": 200.89883337600622,
            "mae_deg": 6.228709557114241,
            "valid_angle_count": 1848
          },
          "user_009": {
            "absolute_error_sum_rad": 209.19962198048597,
            "mae_deg": 6.409762254125185,
            "valid_angle_count": 1870
          },
          "user_010": {
            "absolute_error_sum_rad": 130.93658581505588,
            "mae_deg": 5.500083395201777,
            "valid_angle_count": 1364
          },
          "user_011": {
            "absolute_error_sum_rad": 248.83983815921238,
            "mae_deg": 7.364396953120518,
            "valid_angle_count": 1936
          },
          "user_012": {
            "absolute_error_sum_rad": 132.33688348856958,
            "mae_deg": 5.942276565681463,
            "valid_angle_count": 1276
          },
          "user_013": {
            "absolute_error_sum_rad": 1515.3831926713174,
            "mae_deg": 7.281538182248116,
            "valid_angle_count": 11924
          },
          "user_014": {
            "absolute_error_sum_rad": 116.65398735832423,
            "mae_deg": 6.329338199814808,
            "valid_angle_count": 1056
          },
          "user_015": {
            "absolute_error_sum_rad": 195.00778704564436,
            "mae_deg": 6.4287244936141255,
            "valid_angle_count": 1738
          },
          "user_016": {
            "absolute_error_sum_rad": 150.48792768944986,
            "mae_deg": 4.899047229701929,
            "valid_angle_count": 1760
          },
          "user_017": {
            "absolute_error_sum_rad": 15.94538934994489,
            "mae_deg": 5.190929047981213,
            "valid_angle_count": 176
          },
          "user_018": {
            "absolute_error_sum_rad": 73.94183054566383,
            "mae_deg": 6.419022454148561,
            "valid_angle_count": 660
          },
          "user_019": {
            "absolute_error_sum_rad": 190.8157482589595,
            "mae_deg": 5.0709355472491495,
            "valid_angle_count": 2156
          },
          "user_020": {
            "absolute_error_sum_rad": 187.19235556799686,
            "mae_deg": 6.018704787406541,
            "valid_angle_count": 1782
          },
          "user_021": {
            "absolute_error_sum_rad": 197.10354087754968,
            "mae_deg": 5.2380338679813985,
            "valid_angle_count": 2156
          },
          "user_022": {
            "absolute_error_sum_rad": 101.60363102681004,
            "mae_deg": 8.820392789455573,
            "valid_angle_count": 660
          },
          "user_023": {
            "absolute_error_sum_rad": 167.46899579296587,
            "mae_deg": 5.192243862679158,
            "valid_angle_count": 1848
          },
          "user_024": {
            "absolute_error_sum_rad": 146.1023794527282,
            "mae_deg": 5.284753610770331,
            "valid_angle_count": 1584
          },
          "user_025": {
            "absolute_error_sum_rad": 1162.2510706002904,
            "mae_deg": 7.72171626622891,
            "valid_angle_count": 8624
          },
          "user_026": {
            "absolute_error_sum_rad": 119.23841117217671,
            "mae_deg": 7.393785406938624,
            "valid_angle_count": 924
          },
          "user_027": {
            "absolute_error_sum_rad": 240.28684760950273,
            "mae_deg": 6.953243555822061,
            "valid_angle_count": 1980
          },
          "user_028": {
            "absolute_error_sum_rad": 163.7934287803946,
            "mae_deg": 4.847454638994443,
            "valid_angle_count": 1936
          },
          "user_029": {
            "absolute_error_sum_rad": 117.88601712707896,
            "mae_deg": 6.140337495444148,
            "valid_angle_count": 1100
          },
          "user_030": {
            "absolute_error_sum_rad": 85.2612720602192,
            "mae_deg": 4.441010040879314,
            "valid_angle_count": 1100
          },
          "user_031": {
            "absolute_error_sum_rad": 710.9176090642231,
            "mae_deg": 7.3471462086781,
            "valid_angle_count": 5544
          },
          "user_032": {
            "absolute_error_sum_rad": 59.54116838501068,
            "mae_deg": 7.384107479954285,
            "valid_angle_count": 462
          },
          "user_033": {
            "absolute_error_sum_rad": 61.85803196346387,
            "mae_deg": 5.370006303775492,
            "valid_angle_count": 660
          },
          "user_034": {
            "absolute_error_sum_rad": 93.91232649220547,
            "mae_deg": 5.558656975473145,
            "valid_angle_count": 968
          },
          "user_035": {
            "absolute_error_sum_rad": 93.26441903464729,
            "mae_deg": 5.280294060696511,
            "valid_angle_count": 1012
          },
          "user_036": {
            "absolute_error_sum_rad": 106.25311258505099,
            "mae_deg": 6.289106313276647,
            "valid_angle_count": 968
          },
          "user_037": {
            "absolute_error_sum_rad": 117.26700346419238,
            "mae_deg": 4.925882972613071,
            "valid_angle_count": 1364
          },
          "user_038": {
            "absolute_error_sum_rad": 1126.0450819089456,
            "mae_deg": 6.884083518442773,
            "valid_angle_count": 9372
          },
          "user_039": {
            "absolute_error_sum_rad": 1106.4320118235735,
            "mae_deg": 7.50401096066046,
            "valid_angle_count": 8448
          },
          "user_040": {
            "absolute_error_sum_rad": 222.79197808663594,
            "mae_deg": 10.00394988537254,
            "valid_angle_count": 1276
          }
        },
        "invalid_angle_count": 2816,
        "mode": "center",
        "overall": {
          "absolute_error_sum_rad": 10846.663551143958,
          "mae_deg": 6.761260751979315,
          "valid_angle_count": 91916
        },
        "user_summary": {
          "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
          "mean_mae_deg": 6.662327158460137,
          "std_mae_deg": 1.991625044456936,
          "user_count": 41
        }
      },
      "definition": "all requested labeled centers; vision_valid is the identical crop-present subset",
      "missing_by_hand": {
        "left": 80,
        "right": 93
      },
      "missing_by_split": {
        "both": 9,
        "gesture": 124,
        "user": 40
      },
      "samples": {
        "all": 4306,
        "missing_crop": 173,
        "vision_valid": 4133
      },
      "vision_valid": {
        "by_component": {
          "fingers": {
            "absolute_error_sum_rad": 9509.44344546676,
            "mae_deg": 6.591470783251746,
            "valid_angle_count": 82660
          },
          "wrist": {
            "absolute_error_sum_rad": 1002.05825206177,
            "mae_deg": 6.945766836365333,
            "valid_angle_count": 8266
          }
        },
        "by_hand": {
          "left": {
            "absolute_error_sum_rad": 5426.514251966412,
            "mae_deg": 6.817444285955059,
            "valid_angle_count": 45606
          },
          "right": {
            "absolute_error_sum_rad": 5084.987445562118,
            "mae_deg": 6.42869195736361,
            "valid_angle_count": 45320
          }
        },
        "by_hand_split": {
          "left": {
            "both": {
              "absolute_error_sum_rad": 459.37590200285194,
              "mae_deg": 7.669085196613845,
              "valid_angle_count": 3432
            },
            "gesture": {
              "absolute_error_sum_rad": 2574.5647722930844,
              "mae_deg": 6.0624566634659525,
              "valid_angle_count": 24332
            },
            "user": {
              "absolute_error_sum_rad": 2392.573577670476,
              "mae_deg": 7.683240005326428,
              "valid_angle_count": 17842
            }
          },
          "right": {
            "both": {
              "absolute_error_sum_rad": 427.4149621425404,
              "mae_deg": 7.090061792562659,
              "valid_angle_count": 3454
            },
            "gesture": {
              "absolute_error_sum_rad": 2473.573618340255,
              "mae_deg": 5.888537836371967,
              "valid_angle_count": 24068
            },
            "user": {
              "absolute_error_sum_rad": 2183.998865079323,
              "mae_deg": 7.030785337139393,
              "valid_angle_count": 17798
            }
          }
        },
        "by_split": {
          "both": {
            "absolute_error_sum_rad": 886.7908641453923,
            "mae_deg": 7.3786485370737935,
            "valid_angle_count": 6886
          },
          "gesture": {
            "absolute_error_sum_rad": 5048.138390633339,
            "mae_deg": 5.975971573992853,
            "valid_angle_count": 48400
          },
          "user": {
            "absolute_error_sum_rad": 4576.572442749799,
            "mae_deg": 7.357415421028088,
            "valid_angle_count": 35640
          }
        },
        "by_user": {
          "user_000": {
            "absolute_error_sum_rad": 127.58616805495694,
            "mae_deg": 5.933562462496656,
            "valid_angle_count": 1232
          },
          "user_001": {
            "absolute_error_sum_rad": 168.04711261577904,
            "mae_deg": 5.758606646078822,
            "valid_angle_count": 1672
          },
          "user_002": {
            "absolute_error_sum_rad": 272.225953899906,
            "mae_deg": 10.742009801919785,
            "valid_angle_count": 1452
          },
          "user_003": {
            "absolute_error_sum_rad": 7.097398193203844,
            "mae_deg": 18.484134636107104,
            "valid_angle_count": 22
          },
          "user_004": {
            "absolute_error_sum_rad": 95.53150135820033,
            "mae_deg": 5.07750634357437,
            "valid_angle_count": 1078
          },
          "user_005": {
            "absolute_error_sum_rad": 165.25477151774976,
            "mae_deg": 7.055440351986428,
            "valid_angle_count": 1342
          },
          "user_006": {
            "absolute_error_sum_rad": 134.8748242208676,
            "mae_deg": 4.878635221227629,
            "valid_angle_count": 1584
          },
          "user_007": {
            "absolute_error_sum_rad": 175.69242148043122,
            "mae_deg": 6.355072123271664,
            "valid_angle_count": 1584
          },
          "user_008": {
            "absolute_error_sum_rad": 200.89883337600622,
            "mae_deg": 6.228709557114241,
            "valid_angle_count": 1848
          },
          "user_009": {
            "absolute_error_sum_rad": 209.19962198048597,
            "mae_deg": 6.409762254125185,
            "valid_angle_count": 1870
          },
          "user_010": {
            "absolute_error_sum_rad": 130.93658581505588,
            "mae_deg": 5.500083395201777,
            "valid_angle_count": 1364
          },
          "user_011": {
            "absolute_error_sum_rad": 248.83983815921238,
            "mae_deg": 7.364396953120518,
            "valid_angle_count": 1936
          },
          "user_012": {
            "absolute_error_sum_rad": 132.33688348856958,
            "mae_deg": 5.942276565681463,
            "valid_angle_count": 1276
          },
          "user_013": {
            "absolute_error_sum_rad": 1515.3831926713174,
            "mae_deg": 7.281538182248116,
            "valid_angle_count": 11924
          },
          "user_014": {
            "absolute_error_sum_rad": 116.65398735832423,
            "mae_deg": 6.329338199814808,
            "valid_angle_count": 1056
          },
          "user_015": {
            "absolute_error_sum_rad": 195.00778704564436,
            "mae_deg": 6.4287244936141255,
            "valid_angle_count": 1738
          },
          "user_016": {
            "absolute_error_sum_rad": 150.48792768944986,
            "mae_deg": 4.899047229701929,
            "valid_angle_count": 1760
          },
          "user_017": {
            "absolute_error_sum_rad": 15.94538934994489,
            "mae_deg": 5.190929047981213,
            "valid_angle_count": 176
          },
          "user_018": {
            "absolute_error_sum_rad": 73.94183054566383,
            "mae_deg": 6.419022454148561,
            "valid_angle_count": 660
          },
          "user_019": {
            "absolute_error_sum_rad": 190.8157482589595,
            "mae_deg": 5.0709355472491495,
            "valid_angle_count": 2156
          },
          "user_020": {
            "absolute_error_sum_rad": 187.19235556799686,
            "mae_deg": 6.018704787406541,
            "valid_angle_count": 1782
          },
          "user_021": {
            "absolute_error_sum_rad": 197.10354087754968,
            "mae_deg": 5.2380338679813985,
            "valid_angle_count": 2156
          },
          "user_022": {
            "absolute_error_sum_rad": 101.60363102681004,
            "mae_deg": 8.820392789455573,
            "valid_angle_count": 660
          },
          "user_023": {
            "absolute_error_sum_rad": 167.46899579296587,
            "mae_deg": 5.192243862679158,
            "valid_angle_count": 1848
          },
          "user_024": {
            "absolute_error_sum_rad": 146.1023794527282,
            "mae_deg": 5.284753610770331,
            "valid_angle_count": 1584
          },
          "user_025": {
            "absolute_error_sum_rad": 1140.2459220926357,
            "mae_deg": 7.653617495654979,
            "valid_angle_count": 8536
          },
          "user_026": {
            "absolute_error_sum_rad": 106.9438011207385,
            "mae_deg": 6.962986874209954,
            "valid_angle_count": 880
          },
          "user_027": {
            "absolute_error_sum_rad": 240.28684760950273,
            "mae_deg": 6.953243555822061,
            "valid_angle_count": 1980
          },
          "user_028": {
            "absolute_error_sum_rad": 163.7934287803946,
            "mae_deg": 4.847454638994443,
            "valid_angle_count": 1936
          },
          "user_029": {
            "absolute_error_sum_rad": 117.88601712707896,
            "mae_deg": 6.140337495444148,
            "valid_angle_count": 1100
          },
          "user_030": {
            "absolute_error_sum_rad": 85.2612720602192,
            "mae_deg": 4.441010040879314,
            "valid_angle_count": 1100
          },
          "user_031": {
            "absolute_error_sum_rad": 663.0853736915979,
            "mae_deg": 7.077495039005885,
            "valid_angle_count": 5368
          },
          "user_032": {
            "absolute_error_sum_rad": 46.397886909369845,
            "mae_deg": 6.359816024478907,
            "valid_angle_count": 418
          },
          "user_033": {
            "absolute_error_sum_rad": 61.85803196346387,
            "mae_deg": 5.370006303775492,
            "valid_angle_count": 660
          },
          "user_034": {
            "absolute_error_sum_rad": 93.91232649220547,
            "mae_deg": 5.558656975473145,
            "valid_angle_count": 968
          },
          "user_035": {
            "absolute_error_sum_rad": 93.26441903464729,
            "mae_deg": 5.280294060696511,
            "valid_angle_count": 1012
          },
          "user_036": {
            "absolute_error_sum_rad": 106.25311258505099,
            "mae_deg": 6.289106313276647,
            "valid_angle_count": 968
          },
          "user_037": {
            "absolute_error_sum_rad": 117.26700346419238,
            "mae_deg": 4.925882972613071,
            "valid_angle_count": 1364
          },
          "user_038": {
            "absolute_error_sum_rad": 1126.0450819089456,
            "mae_deg": 6.884083518442773,
            "valid_angle_count": 9372
          },
          "user_039": {
            "absolute_error_sum_rad": 1106.4320118235735,
            "mae_deg": 7.50401096066046,
            "valid_angle_count": 8448
          },
          "user_040": {
            "absolute_error_sum_rad": 116.34048106713453,
            "mae_deg": 6.312328173928474,
            "valid_angle_count": 1056
          }
        },
        "invalid_angle_count": 0,
        "mode": "center",
        "overall": {
          "absolute_error_sum_rad": 10511.50169752853,
          "mae_deg": 6.623679515352982,
          "valid_angle_count": 90926
        },
        "user_summary": {
          "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
          "mean_mae_deg": 6.4991266055686046,
          "std_mae_deg": 2.2280884774754397,
          "user_count": 41
        }
      }
    },
    "unique_rule": "first original repetition only; duplicated consumptions remain in native and consumed metrics",
    "unique_samples": 4306
  },
  "training": {
    "completed_epochs": 94,
    "fit_result_sha256": "c37b65cfce6bbdb9ffb893a586d29845cd8ceee2cb4ebd8eda5f274ab3a74569",
    "full_training_terminated": true,
    "optimizer_executions": 52828,
    "scope": "full",
    "selected_best": {
      "checkpoint_sha256": "77b5a83c7c279568be7c4a7c2bbf304d95c4672aee2e827f106418022c0a3a4b",
      "file": "vit-small-epoch=063-val_mae=0.1074.ckpt",
      "score": 0.107392817735672,
      "validated": true
    }
  },
  "validation_profile_sha256": "306bf7a9407a8a347bb8fdbc04993f007a99b3324e32f9124cbc5ba16a781689",
  "variant": "vit_base"
}
```

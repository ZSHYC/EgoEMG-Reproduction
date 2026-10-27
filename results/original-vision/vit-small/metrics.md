# DINOv2 视觉中心评价

完整训练实验

模型：vit_small；训练完成 124 轮，实际更新 69688 次。

验证选择：vit-small-epoch=093-val_mae=0.1012.ckpt；不以最后一轮替代验证最优。

测试消费 8612 手样本，唯一 4306，重复消费 4306。

| 统计人口 | MAE (°) | 有效角度数 | 弧度绝对误差和 |
| --- | ---: | ---: | ---: |
| 全部重复消费 | 6.464179 | 183832 | 20740.14753 |
| 唯一首遍 | 6.464179 | 91916 | 10370.07377 |
| 重复消费中的有图子集 | 6.324737 | 181852 | 20074.18543 |

原生按样本数加权的批量 MAE：0.112734586 rad。

该指标与按有效角度数汇总的 pooled MAE 分开报告。缺图仍保留原生成标签 mask；不增加腕角过滤。

采用可用旧版 EgoEMG 队列、独立 train 留出验证、单 rank/workers0；未证明与论文原表或缺失的合并训练来源数值等价。
当前报告只验收所列实验，整篇论文复现仍未完成。

## 完整公开证据

以下与 metrics.json 一致，包含逐手、逐条件、匿名逐用户、逐角、指组、用户交集与来源摘要。

```json
{
  "cohort_sha256": "4580be4fcf98e1f5e683477db12bd2d2020a61cfd0a79755a2cea1171b2cecab",
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
  "independent_test_sha256": "f7d3c356affcc20b25dae3807cde765773ac761eb54bee5381c50dd5f195ef5e",
  "native_metrics": {
    "test_derivatives/acc": 0.0,
    "test_derivatives/jerk": 0.0,
    "test_derivatives/vel": 0.0,
    "test_landmark/all": 9.077659606933594,
    "test_landmark/fingertip": 14.45954704284668,
    "test_loss": 0.4019254744052887,
    "test_mae": 0.1127345860004425,
    "test_mae_per_finger/index": 0.0980416089296341,
    "test_mae_per_finger/middle": 0.10449723154306412,
    "test_mae_per_finger/pinky": 0.14720724523067474,
    "test_mae_per_finger/ring": 0.10732729732990265,
    "test_mae_per_finger/thumb": 0.10389866679906845,
    "test_mae_per_pd/distal": 0.08772513270378113,
    "test_mae_per_pd/mid": 0.13770268857479095,
    "test_mae_per_pd/proximal": 0.11167491227388382,
    "test_mae_wrist/deviation": 0.1267305612564087,
    "test_mae_wrist/flexion": 0.10954222083091736
  },
  "native_test_sha256": "b4b5717b522b7d99a465bd4477b60547d51def2fdb11ecb032a3904b4d4a8eda",
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
  "prediction_sha256": "2c3d8332ca0f2a92bd0fa6227ae0abf8a39acd8a64a2f6e64c99d4b8d7f7f2b1",
  "scientific_eligible": true,
  "scope": "available legacy cohort; single rank/workers0; original vision semantics; independent train-only validation",
  "sha256": "a3a75211494e27a800ecb40e000556adf36499b0ae5936b0484b4be0279602a5",
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
            "absolute_error_sum_rad": 18760.8826392021,
            "mae_deg": 6.432021273136295,
            "valid_angle_count": 167120
          },
          "wrist": {
            "absolute_error_sum_rad": 1979.2648916846265,
            "mae_deg": 6.785754238388411,
            "valid_angle_count": 16712
          }
        },
        "by_finger": {
          "index": {
            "absolute_error_sum_rad": 3278.4407142572454,
            "mae_deg": 5.619938257264099,
            "valid_angle_count": 33424
          },
          "middle": {
            "absolute_error_sum_rad": 3491.612995628122,
            "mae_deg": 5.985360469797802,
            "valid_angle_count": 33424
          },
          "pinky": {
            "absolute_error_sum_rad": 4921.696679161891,
            "mae_deg": 8.43682526805676,
            "valid_angle_count": 33424
          },
          "ring": {
            "absolute_error_sum_rad": 3593.2167709600576,
            "mae_deg": 6.159530751903933,
            "valid_angle_count": 33424
          },
          "thumb": {
            "absolute_error_sum_rad": 3475.915479194784,
            "mae_deg": 5.9584516186588745,
            "valid_angle_count": 33424
          }
        },
        "by_hand": {
          "left": {
            "absolute_error_sum_rad": 10670.632310797819,
            "mae_deg": 6.629317706287344,
            "valid_angle_count": 92224
          },
          "right": {
            "absolute_error_sum_rad": 10069.515220088908,
            "mae_deg": 6.297929480545811,
            "valid_angle_count": 91608
          }
        },
        "by_hand_split": {
          "left": {
            "both": {
              "absolute_error_sum_rad": 895.3290846046875,
              "mae_deg": 7.473569030180653,
              "valid_angle_count": 6864
            },
            "gesture": {
              "absolute_error_sum_rad": 5117.459604596945,
              "mae_deg": 5.9392488488918325,
              "valid_angle_count": 49368
            },
            "user": {
              "absolute_error_sum_rad": 4657.8436215961865,
              "mae_deg": 7.414836106617914,
              "valid_angle_count": 35992
            }
          },
          "right": {
            "both": {
              "absolute_error_sum_rad": 846.1601683884328,
              "mae_deg": 7.018153798601079,
              "valid_angle_count": 6908
            },
            "gesture": {
              "absolute_error_sum_rad": 4942.022927183323,
              "mae_deg": 5.797646518929021,
              "valid_angle_count": 48840
            },
            "user": {
              "absolute_error_sum_rad": 4281.332124517152,
              "mae_deg": 6.840553860251283,
              "valid_angle_count": 35860
            }
          }
        },
        "by_joint": [
          {
            "absolute_error_sum_rad": 675.9579776785795,
            "index": 0,
            "mae_deg": 4.634937679413704,
            "name": "THUMB_CMC_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 821.6558265405401,
            "index": 1,
            "mae_deg": 5.633964944124725,
            "name": "THUMB_CMC_AA",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 976.2123672235757,
            "index": 2,
            "mae_deg": 6.693734867207533,
            "name": "THUMB_MCP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1002.0893077520886,
            "index": 3,
            "mae_deg": 6.871168983889537,
            "name": "THUMB_IP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 721.4259380178992,
            "index": 4,
            "mae_deg": 4.946704341753488,
            "name": "INDEX_MCP_AA",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 814.8766924862866,
            "index": 5,
            "mae_deg": 5.587481486721407,
            "name": "INDEX_MCP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1135.735573964077,
            "index": 6,
            "mae_deg": 7.787560439326204,
            "name": "INDEX_PIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 606.4025097889826,
            "index": 7,
            "mae_deg": 4.158006761255301,
            "name": "INDEX_DIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 711.5296481100959,
            "index": 8,
            "mae_deg": 4.878847036277778,
            "name": "MIDDLE_MCP_AA",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 993.3293561989267,
            "index": 9,
            "mae_deg": 6.811103372025577,
            "name": "MIDDLE_MCP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1057.425646974938,
            "index": 10,
            "mae_deg": 7.250601570195605,
            "name": "MIDDLE_PIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 729.3283443441615,
            "index": 11,
            "mae_deg": 5.000889900692251,
            "name": "MIDDLE_DIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 643.9211267792666,
            "index": 12,
            "mae_deg": 4.415266024863618,
            "name": "RING_MCP_AA",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1149.444882766751,
            "index": 13,
            "mae_deg": 7.881563016448604,
            "name": "RING_MCP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1224.2206649072468,
            "index": 14,
            "mae_deg": 8.394288809464417,
            "name": "RING_PIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 575.630096506793,
            "index": 15,
            "mae_deg": 3.9470051568390994,
            "name": "RING_DIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1118.129958179663,
            "index": 16,
            "mae_deg": 7.666841497227613,
            "name": "PINKY_MCP_AA",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1683.8928259170207,
            "index": 17,
            "mae_deg": 11.546188616252122,
            "name": "PINKY_MCP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1366.1282092654146,
            "index": 18,
            "mae_deg": 9.367326551540591,
            "name": "PINKY_PIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 753.5456857997924,
            "index": 19,
            "mae_deg": 5.166944407206716,
            "name": "PINKY_DIP_FE",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 916.6334476365446,
            "index": 20,
            "mae_deg": 6.285211573731445,
            "name": "wrist_pitch",
            "valid_angle_count": 8356
          },
          {
            "absolute_error_sum_rad": 1062.631444048082,
            "index": 21,
            "mae_deg": 7.286296903045378,
            "name": "wrist_yaw",
            "valid_angle_count": 8356
          }
        ],
        "by_pd_group": {
          "distal": {
            "absolute_error_sum_rad": 3666.995944191818,
            "mae_deg": 5.028803041976581,
            "valid_angle_count": 41780
          },
          "mid": {
            "absolute_error_sum_rad": 5759.722462335252,
            "mae_deg": 7.89870244754687,
            "valid_angle_count": 41780
          },
          "proximal": {
            "absolute_error_sum_rad": 9334.16423267503,
            "mae_deg": 6.400289801510863,
            "valid_angle_count": 83560
          }
        },
        "by_split": {
          "both": {
            "absolute_error_sum_rad": 1741.4892529931203,
            "mae_deg": 7.245133914020932,
            "valid_angle_count": 13772
          },
          "gesture": {
            "absolute_error_sum_rad": 10059.482531780268,
            "mae_deg": 5.868828335335058,
            "valid_angle_count": 98208
          },
          "user": {
            "absolute_error_sum_rad": 8939.175746113338,
            "mae_deg": 7.1282224931526335,
            "valid_angle_count": 71852
          }
        },
        "by_split_user_summary": {
          "both": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 7.2293140778439335,
            "std_mae_deg": 0.5286218693903406,
            "user_count": 5
          },
          "gesture": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 5.909170092619415,
            "std_mae_deg": 1.3926444115973964,
            "user_count": 36
          },
          "user": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 8.649698742054984,
            "std_mae_deg": 3.2728336055022367,
            "user_count": 6
          }
        },
        "by_user": {
          "user_000": {
            "absolute_error_sum_rad": 245.10766620188951,
            "mae_deg": 5.699527110255537,
            "valid_angle_count": 2464
          },
          "user_001": {
            "absolute_error_sum_rad": 366.99698724783957,
            "mae_deg": 6.126858526607722,
            "valid_angle_count": 3432
          },
          "user_002": {
            "absolute_error_sum_rad": 539.5943683211226,
            "mae_deg": 10.646170782998613,
            "valid_angle_count": 2904
          },
          "user_003": {
            "absolute_error_sum_rad": 24.473325117258355,
            "mae_deg": 15.934298180345618,
            "valid_angle_count": 88
          },
          "user_004": {
            "absolute_error_sum_rad": 355.00729911460076,
            "mae_deg": 7.337813829583345,
            "valid_angle_count": 2772
          },
          "user_005": {
            "absolute_error_sum_rad": 315.01744496109313,
            "mae_deg": 6.724728043690513,
            "valid_angle_count": 2684
          },
          "user_006": {
            "absolute_error_sum_rad": 285.69149378611473,
            "mae_deg": 5.027308610790036,
            "valid_angle_count": 3256
          },
          "user_007": {
            "absolute_error_sum_rad": 325.46317246207036,
            "mae_deg": 5.886258260421424,
            "valid_angle_count": 3168
          },
          "user_008": {
            "absolute_error_sum_rad": 350.81146079977043,
            "mae_deg": 5.438316046711572,
            "valid_angle_count": 3696
          },
          "user_009": {
            "absolute_error_sum_rad": 423.4536486052675,
            "mae_deg": 6.487194354143756,
            "valid_angle_count": 3740
          },
          "user_010": {
            "absolute_error_sum_rad": 251.14929400084657,
            "mae_deg": 5.274851383408649,
            "valid_angle_count": 2728
          },
          "user_011": {
            "absolute_error_sum_rad": 480.6193831748096,
            "mae_deg": 7.111947884322706,
            "valid_angle_count": 3872
          },
          "user_012": {
            "absolute_error_sum_rad": 246.66447192149644,
            "mae_deg": 5.537944042682189,
            "valid_angle_count": 2552
          },
          "user_013": {
            "absolute_error_sum_rad": 2896.6716658593505,
            "mae_deg": 6.959370223451451,
            "valid_angle_count": 23848
          },
          "user_014": {
            "absolute_error_sum_rad": 239.74348559882492,
            "mae_deg": 6.503925137579593,
            "valid_angle_count": 2112
          },
          "user_015": {
            "absolute_error_sum_rad": 382.42885349941207,
            "mae_deg": 6.303670675933019,
            "valid_angle_count": 3476
          },
          "user_016": {
            "absolute_error_sum_rad": 308.89885922009125,
            "mae_deg": 5.028011627760513,
            "valid_angle_count": 3520
          },
          "user_017": {
            "absolute_error_sum_rad": 30.43305948190391,
            "mae_deg": 4.953653028362743,
            "valid_angle_count": 352
          },
          "user_018": {
            "absolute_error_sum_rad": 130.64628744870424,
            "mae_deg": 5.670818848381615,
            "valid_angle_count": 1320
          },
          "user_019": {
            "absolute_error_sum_rad": 342.1457488993183,
            "mae_deg": 4.546267947651616,
            "valid_angle_count": 4312
          },
          "user_020": {
            "absolute_error_sum_rad": 330.6866448953515,
            "mae_deg": 5.316203449451462,
            "valid_angle_count": 3564
          },
          "user_021": {
            "absolute_error_sum_rad": 379.5851429172908,
            "mae_deg": 5.043744586046135,
            "valid_angle_count": 4312
          },
          "user_022": {
            "absolute_error_sum_rad": 192.2783523364924,
            "mae_deg": 8.346013697432141,
            "valid_angle_count": 1320
          },
          "user_023": {
            "absolute_error_sum_rad": 321.09500453423243,
            "mae_deg": 4.977648426013407,
            "valid_angle_count": 3696
          },
          "user_024": {
            "absolute_error_sum_rad": 278.1000771081308,
            "mae_deg": 5.0296593120450295,
            "valid_angle_count": 3168
          },
          "user_025": {
            "absolute_error_sum_rad": 2199.9983234110914,
            "mae_deg": 7.308129572548276,
            "valid_angle_count": 17248
          },
          "user_026": {
            "absolute_error_sum_rad": 217.60291060223244,
            "mae_deg": 6.746606270168005,
            "valid_angle_count": 1848
          },
          "user_027": {
            "absolute_error_sum_rad": 456.62724262301344,
            "mae_deg": 6.606771164897708,
            "valid_angle_count": 3960
          },
          "user_028": {
            "absolute_error_sum_rad": 300.9443483238574,
            "mae_deg": 4.453213075225202,
            "valid_angle_count": 3872
          },
          "user_029": {
            "absolute_error_sum_rad": 200.69411009945907,
            "mae_deg": 5.226784309924034,
            "valid_angle_count": 2200
          },
          "user_030": {
            "absolute_error_sum_rad": 163.48937625344843,
            "mae_deg": 4.257841479340427,
            "valid_angle_count": 2200
          },
          "user_031": {
            "absolute_error_sum_rad": 1330.804768876009,
            "mae_deg": 6.876758352496238,
            "valid_angle_count": 11088
          },
          "user_032": {
            "absolute_error_sum_rad": 109.33215051703155,
            "mae_deg": 6.779513841682867,
            "valid_angle_count": 924
          },
          "user_033": {
            "absolute_error_sum_rad": 114.12934173922986,
            "mae_deg": 4.953886060806164,
            "valid_angle_count": 1320
          },
          "user_034": {
            "absolute_error_sum_rad": 179.4954407598343,
            "mae_deg": 5.312154544100721,
            "valid_angle_count": 1936
          },
          "user_035": {
            "absolute_error_sum_rad": 169.5331357588293,
            "mae_deg": 4.799176465711135,
            "valid_angle_count": 2024
          },
          "user_036": {
            "absolute_error_sum_rad": 196.9739586529322,
            "mae_deg": 5.829430012808573,
            "valid_angle_count": 1936
          },
          "user_037": {
            "absolute_error_sum_rad": 220.41026176378364,
            "mae_deg": 4.6292440470815635,
            "valid_angle_count": 2728
          },
          "user_038": {
            "absolute_error_sum_rad": 2179.4351058437314,
            "mae_deg": 6.661994947049385,
            "valid_angle_count": 18744
          },
          "user_039": {
            "absolute_error_sum_rad": 2223.9061868946224,
            "mae_deg": 7.541455879622036,
            "valid_angle_count": 16896
          },
          "user_040": {
            "absolute_error_sum_rad": 434.00767125433777,
            "mae_deg": 9.744046958924315,
            "valid_angle_count": 2552
          }
        },
        "invalid_angle_count": 5632,
        "mode": "center",
        "overall": {
          "absolute_error_sum_rad": 20740.14753088673,
          "mae_deg": 6.464178815431943,
          "valid_angle_count": 183832
        },
        "user_summary": {
          "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
          "mean_mae_deg": 6.332663682889197,
          "std_mae_deg": 2.023265587836175,
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
            "absolute_error_sum_rad": 18142.238469001182,
            "mae_deg": 6.287646353699806,
            "valid_angle_count": 165320
          },
          "wrist": {
            "absolute_error_sum_rad": 1931.9469560669859,
            "mae_deg": 6.695645222948498,
            "valid_angle_count": 16532
          }
        },
        "by_hand": {
          "left": {
            "absolute_error_sum_rad": 10324.3565435008,
            "mae_deg": 6.485353419844654,
            "valid_angle_count": 91212
          },
          "right": {
            "absolute_error_sum_rad": 9749.828881567368,
            "mae_deg": 6.163107302389298,
            "valid_angle_count": 90640
          }
        },
        "by_hand_split": {
          "left": {
            "both": {
              "absolute_error_sum_rad": 895.3290846046875,
              "mae_deg": 7.473569030180653,
              "valid_angle_count": 6864
            },
            "gesture": {
              "absolute_error_sum_rad": 4852.88956136605,
              "mae_deg": 5.713671097923886,
              "valid_angle_count": 48664
            },
            "user": {
              "absolute_error_sum_rad": 4576.137897530063,
              "mae_deg": 7.347645667479615,
              "valid_angle_count": 35684
            }
          },
          "right": {
            "both": {
              "absolute_error_sum_rad": 846.1601683884328,
              "mae_deg": 7.018153798601079,
              "valid_angle_count": 6908
            },
            "gesture": {
              "absolute_error_sum_rad": 4689.792211642154,
              "mae_deg": 5.58221083015663,
              "valid_angle_count": 48136
            },
            "user": {
              "absolute_error_sum_rad": 4213.876501536781,
              "mae_deg": 6.782709824907577,
              "valid_angle_count": 35596
            }
          }
        },
        "by_split": {
          "both": {
            "absolute_error_sum_rad": 1741.4892529931203,
            "mae_deg": 7.245133914020932,
            "valid_angle_count": 13772
          },
          "gesture": {
            "absolute_error_sum_rad": 9542.681773008204,
            "mae_deg": 5.648299492043259,
            "valid_angle_count": 96800
          },
          "user": {
            "absolute_error_sum_rad": 8790.014399066844,
            "mae_deg": 7.065526472022344,
            "valid_angle_count": 71280
          }
        },
        "by_user": {
          "user_000": {
            "absolute_error_sum_rad": 245.10766620188951,
            "mae_deg": 5.699527110255537,
            "valid_angle_count": 2464
          },
          "user_001": {
            "absolute_error_sum_rad": 317.34922246821225,
            "mae_deg": 5.437431542819019,
            "valid_angle_count": 3344
          },
          "user_002": {
            "absolute_error_sum_rad": 539.5943683211226,
            "mae_deg": 10.646170782998613,
            "valid_angle_count": 2904
          },
          "user_003": {
            "absolute_error_sum_rad": 14.299638506723568,
            "mae_deg": 18.620657613591256,
            "valid_angle_count": 44
          },
          "user_004": {
            "absolute_error_sum_rad": 174.27877866348717,
            "mae_deg": 4.631464970367549,
            "valid_angle_count": 2156
          },
          "user_005": {
            "absolute_error_sum_rad": 315.01744496109313,
            "mae_deg": 6.724728043690513,
            "valid_angle_count": 2684
          },
          "user_006": {
            "absolute_error_sum_rad": 261.69710569566814,
            "mae_deg": 4.7329986323077025,
            "valid_angle_count": 3168
          },
          "user_007": {
            "absolute_error_sum_rad": 325.46317246207036,
            "mae_deg": 5.886258260421424,
            "valid_angle_count": 3168
          },
          "user_008": {
            "absolute_error_sum_rad": 350.81146079977043,
            "mae_deg": 5.438316046711572,
            "valid_angle_count": 3696
          },
          "user_009": {
            "absolute_error_sum_rad": 423.4536486052675,
            "mae_deg": 6.487194354143756,
            "valid_angle_count": 3740
          },
          "user_010": {
            "absolute_error_sum_rad": 251.14929400084657,
            "mae_deg": 5.274851383408649,
            "valid_angle_count": 2728
          },
          "user_011": {
            "absolute_error_sum_rad": 480.6193831748096,
            "mae_deg": 7.111947884322706,
            "valid_angle_count": 3872
          },
          "user_012": {
            "absolute_error_sum_rad": 246.66447192149644,
            "mae_deg": 5.537944042682189,
            "valid_angle_count": 2552
          },
          "user_013": {
            "absolute_error_sum_rad": 2896.6716658593505,
            "mae_deg": 6.959370223451451,
            "valid_angle_count": 23848
          },
          "user_014": {
            "absolute_error_sum_rad": 239.74348559882492,
            "mae_deg": 6.503925137579593,
            "valid_angle_count": 2112
          },
          "user_015": {
            "absolute_error_sum_rad": 382.42885349941207,
            "mae_deg": 6.303670675933019,
            "valid_angle_count": 3476
          },
          "user_016": {
            "absolute_error_sum_rad": 308.89885922009125,
            "mae_deg": 5.028011627760513,
            "valid_angle_count": 3520
          },
          "user_017": {
            "absolute_error_sum_rad": 30.43305948190391,
            "mae_deg": 4.953653028362743,
            "valid_angle_count": 352
          },
          "user_018": {
            "absolute_error_sum_rad": 130.64628744870424,
            "mae_deg": 5.670818848381615,
            "valid_angle_count": 1320
          },
          "user_019": {
            "absolute_error_sum_rad": 342.1457488993183,
            "mae_deg": 4.546267947651616,
            "valid_angle_count": 4312
          },
          "user_020": {
            "absolute_error_sum_rad": 330.6866448953515,
            "mae_deg": 5.316203449451462,
            "valid_angle_count": 3564
          },
          "user_021": {
            "absolute_error_sum_rad": 379.5851429172908,
            "mae_deg": 5.043744586046135,
            "valid_angle_count": 4312
          },
          "user_022": {
            "absolute_error_sum_rad": 192.2783523364924,
            "mae_deg": 8.346013697432141,
            "valid_angle_count": 1320
          },
          "user_023": {
            "absolute_error_sum_rad": 321.09500453423243,
            "mae_deg": 4.977648426013407,
            "valid_angle_count": 3696
          },
          "user_024": {
            "absolute_error_sum_rad": 278.1000771081308,
            "mae_deg": 5.0296593120450295,
            "valid_angle_count": 3168
          },
          "user_025": {
            "absolute_error_sum_rad": 2156.696766630157,
            "mae_deg": 7.238145643007207,
            "valid_angle_count": 17072
          },
          "user_026": {
            "absolute_error_sum_rad": 193.56434967904352,
            "mae_deg": 6.301375170911162,
            "valid_angle_count": 1760
          },
          "user_027": {
            "absolute_error_sum_rad": 456.62724262301344,
            "mae_deg": 6.606771164897708,
            "valid_angle_count": 3960
          },
          "user_028": {
            "absolute_error_sum_rad": 300.9443483238574,
            "mae_deg": 4.453213075225202,
            "valid_angle_count": 3872
          },
          "user_029": {
            "absolute_error_sum_rad": 200.69411009945907,
            "mae_deg": 5.226784309924034,
            "valid_angle_count": 2200
          },
          "user_030": {
            "absolute_error_sum_rad": 163.48937625344843,
            "mae_deg": 4.257841479340427,
            "valid_angle_count": 2200
          },
          "user_031": {
            "absolute_error_sum_rad": 1235.1186652209835,
            "mae_deg": 6.5915691798615885,
            "valid_angle_count": 10736
          },
          "user_032": {
            "absolute_error_sum_rad": 83.44268229231238,
            "mae_deg": 5.71879608445037,
            "valid_angle_count": 836
          },
          "user_033": {
            "absolute_error_sum_rad": 114.12934173922986,
            "mae_deg": 4.953886060806164,
            "valid_angle_count": 1320
          },
          "user_034": {
            "absolute_error_sum_rad": 179.4954407598343,
            "mae_deg": 5.312154544100721,
            "valid_angle_count": 1936
          },
          "user_035": {
            "absolute_error_sum_rad": 169.5331357588293,
            "mae_deg": 4.799176465711135,
            "valid_angle_count": 2024
          },
          "user_036": {
            "absolute_error_sum_rad": 196.9739586529322,
            "mae_deg": 5.829430012808573,
            "valid_angle_count": 1936
          },
          "user_037": {
            "absolute_error_sum_rad": 220.41026176378364,
            "mae_deg": 4.6292440470815635,
            "valid_angle_count": 2728
          },
          "user_038": {
            "absolute_error_sum_rad": 2179.4351058437314,
            "mae_deg": 6.661994947049385,
            "valid_angle_count": 18744
          },
          "user_039": {
            "absolute_error_sum_rad": 2223.9061868946224,
            "mae_deg": 7.541455879622036,
            "valid_angle_count": 16896
          },
          "user_040": {
            "absolute_error_sum_rad": 221.50561495136935,
            "mae_deg": 6.009155717406898,
            "valid_angle_count": 2112
          }
        },
        "invalid_angle_count": 0,
        "mode": "center",
        "overall": {
          "absolute_error_sum_rad": 20074.18542506817,
          "mae_deg": 6.324737159995143,
          "valid_angle_count": 181852
        },
        "user_summary": {
          "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
          "mean_mae_deg": 6.1716944246349605,
          "std_mae_deg": 2.3015894965943495,
          "user_count": 41
        }
      }
    },
    "consumed_samples": 8612,
    "duplicate_consumptions": 4306,
    "format": "original_vision_center_statistics_v1",
    "native_batch_weighted_mae_rad": 0.1127345860004425,
    "role": "test",
    "unique_first_repeat": {
      "all": {
        "by_component": {
          "fingers": {
            "absolute_error_sum_rad": 9380.44131960105,
            "mae_deg": 6.432021273136295,
            "valid_angle_count": 83560
          },
          "wrist": {
            "absolute_error_sum_rad": 989.6324458423132,
            "mae_deg": 6.785754238388411,
            "valid_angle_count": 8356
          }
        },
        "by_finger": {
          "index": {
            "absolute_error_sum_rad": 1639.2203571286227,
            "mae_deg": 5.619938257264099,
            "valid_angle_count": 16712
          },
          "middle": {
            "absolute_error_sum_rad": 1745.806497814061,
            "mae_deg": 5.985360469797802,
            "valid_angle_count": 16712
          },
          "pinky": {
            "absolute_error_sum_rad": 2460.8483395809453,
            "mae_deg": 8.43682526805676,
            "valid_angle_count": 16712
          },
          "ring": {
            "absolute_error_sum_rad": 1796.6083854800288,
            "mae_deg": 6.159530751903933,
            "valid_angle_count": 16712
          },
          "thumb": {
            "absolute_error_sum_rad": 1737.957739597392,
            "mae_deg": 5.9584516186588745,
            "valid_angle_count": 16712
          }
        },
        "by_hand": {
          "left": {
            "absolute_error_sum_rad": 5335.316155398909,
            "mae_deg": 6.629317706287344,
            "valid_angle_count": 46112
          },
          "right": {
            "absolute_error_sum_rad": 5034.757610044454,
            "mae_deg": 6.297929480545811,
            "valid_angle_count": 45804
          }
        },
        "by_hand_split": {
          "left": {
            "both": {
              "absolute_error_sum_rad": 447.66454230234376,
              "mae_deg": 7.473569030180653,
              "valid_angle_count": 3432
            },
            "gesture": {
              "absolute_error_sum_rad": 2558.7298022984724,
              "mae_deg": 5.9392488488918325,
              "valid_angle_count": 24684
            },
            "user": {
              "absolute_error_sum_rad": 2328.9218107980932,
              "mae_deg": 7.414836106617914,
              "valid_angle_count": 17996
            }
          },
          "right": {
            "both": {
              "absolute_error_sum_rad": 423.0800841942164,
              "mae_deg": 7.018153798601079,
              "valid_angle_count": 3454
            },
            "gesture": {
              "absolute_error_sum_rad": 2471.0114635916616,
              "mae_deg": 5.797646518929021,
              "valid_angle_count": 24420
            },
            "user": {
              "absolute_error_sum_rad": 2140.666062258576,
              "mae_deg": 6.840553860251283,
              "valid_angle_count": 17930
            }
          }
        },
        "by_joint": [
          {
            "absolute_error_sum_rad": 337.97898883928974,
            "index": 0,
            "mae_deg": 4.634937679413704,
            "name": "THUMB_CMC_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 410.8279132702701,
            "index": 1,
            "mae_deg": 5.633964944124725,
            "name": "THUMB_CMC_AA",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 488.10618361178786,
            "index": 2,
            "mae_deg": 6.693734867207533,
            "name": "THUMB_MCP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 501.0446538760443,
            "index": 3,
            "mae_deg": 6.871168983889537,
            "name": "THUMB_IP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 360.7129690089496,
            "index": 4,
            "mae_deg": 4.946704341753488,
            "name": "INDEX_MCP_AA",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 407.4383462431433,
            "index": 5,
            "mae_deg": 5.587481486721407,
            "name": "INDEX_MCP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 567.8677869820385,
            "index": 6,
            "mae_deg": 7.787560439326204,
            "name": "INDEX_PIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 303.2012548944913,
            "index": 7,
            "mae_deg": 4.158006761255301,
            "name": "INDEX_DIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 355.76482405504794,
            "index": 8,
            "mae_deg": 4.878847036277778,
            "name": "MIDDLE_MCP_AA",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 496.66467809946334,
            "index": 9,
            "mae_deg": 6.811103372025577,
            "name": "MIDDLE_MCP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 528.712823487469,
            "index": 10,
            "mae_deg": 7.250601570195605,
            "name": "MIDDLE_PIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 364.6641721720807,
            "index": 11,
            "mae_deg": 5.000889900692251,
            "name": "MIDDLE_DIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 321.9605633896333,
            "index": 12,
            "mae_deg": 4.415266024863618,
            "name": "RING_MCP_AA",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 574.7224413833756,
            "index": 13,
            "mae_deg": 7.881563016448604,
            "name": "RING_MCP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 612.1103324536234,
            "index": 14,
            "mae_deg": 8.394288809464417,
            "name": "RING_PIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 287.8150482533965,
            "index": 15,
            "mae_deg": 3.9470051568390994,
            "name": "RING_DIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 559.0649790898315,
            "index": 16,
            "mae_deg": 7.666841497227613,
            "name": "PINKY_MCP_AA",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 841.9464129585103,
            "index": 17,
            "mae_deg": 11.546188616252122,
            "name": "PINKY_MCP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 683.0641046327073,
            "index": 18,
            "mae_deg": 9.367326551540591,
            "name": "PINKY_PIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 376.7728428998962,
            "index": 19,
            "mae_deg": 5.166944407206716,
            "name": "PINKY_DIP_FE",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 458.3167238182723,
            "index": 20,
            "mae_deg": 6.285211573731445,
            "name": "wrist_pitch",
            "valid_angle_count": 4178
          },
          {
            "absolute_error_sum_rad": 531.315722024041,
            "index": 21,
            "mae_deg": 7.286296903045378,
            "name": "wrist_yaw",
            "valid_angle_count": 4178
          }
        ],
        "by_pd_group": {
          "distal": {
            "absolute_error_sum_rad": 1833.497972095909,
            "mae_deg": 5.028803041976581,
            "valid_angle_count": 20890
          },
          "mid": {
            "absolute_error_sum_rad": 2879.861231167626,
            "mae_deg": 7.89870244754687,
            "valid_angle_count": 20890
          },
          "proximal": {
            "absolute_error_sum_rad": 4667.082116337515,
            "mae_deg": 6.400289801510863,
            "valid_angle_count": 41780
          }
        },
        "by_split": {
          "both": {
            "absolute_error_sum_rad": 870.7446264965602,
            "mae_deg": 7.245133914020932,
            "valid_angle_count": 6886
          },
          "gesture": {
            "absolute_error_sum_rad": 5029.741265890134,
            "mae_deg": 5.868828335335058,
            "valid_angle_count": 49104
          },
          "user": {
            "absolute_error_sum_rad": 4469.587873056669,
            "mae_deg": 7.1282224931526335,
            "valid_angle_count": 35926
          }
        },
        "by_split_user_summary": {
          "both": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 7.2293140778439335,
            "std_mae_deg": 0.5286218693903406,
            "user_count": 5
          },
          "gesture": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 5.909170092619415,
            "std_mae_deg": 1.3926444115973964,
            "user_count": 36
          },
          "user": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 8.649698742054984,
            "std_mae_deg": 3.2728336055022367,
            "user_count": 6
          }
        },
        "by_user": {
          "user_000": {
            "absolute_error_sum_rad": 122.55383310094476,
            "mae_deg": 5.699527110255537,
            "valid_angle_count": 1232
          },
          "user_001": {
            "absolute_error_sum_rad": 183.49849362391979,
            "mae_deg": 6.126858526607722,
            "valid_angle_count": 1716
          },
          "user_002": {
            "absolute_error_sum_rad": 269.7971841605613,
            "mae_deg": 10.646170782998613,
            "valid_angle_count": 1452
          },
          "user_003": {
            "absolute_error_sum_rad": 12.236662558629178,
            "mae_deg": 15.934298180345618,
            "valid_angle_count": 44
          },
          "user_004": {
            "absolute_error_sum_rad": 177.50364955730038,
            "mae_deg": 7.337813829583345,
            "valid_angle_count": 1386
          },
          "user_005": {
            "absolute_error_sum_rad": 157.50872248054657,
            "mae_deg": 6.724728043690513,
            "valid_angle_count": 1342
          },
          "user_006": {
            "absolute_error_sum_rad": 142.84574689305737,
            "mae_deg": 5.027308610790036,
            "valid_angle_count": 1628
          },
          "user_007": {
            "absolute_error_sum_rad": 162.73158623103518,
            "mae_deg": 5.886258260421424,
            "valid_angle_count": 1584
          },
          "user_008": {
            "absolute_error_sum_rad": 175.40573039988521,
            "mae_deg": 5.438316046711572,
            "valid_angle_count": 1848
          },
          "user_009": {
            "absolute_error_sum_rad": 211.72682430263376,
            "mae_deg": 6.487194354143756,
            "valid_angle_count": 1870
          },
          "user_010": {
            "absolute_error_sum_rad": 125.57464700042328,
            "mae_deg": 5.274851383408649,
            "valid_angle_count": 1364
          },
          "user_011": {
            "absolute_error_sum_rad": 240.3096915874048,
            "mae_deg": 7.111947884322706,
            "valid_angle_count": 1936
          },
          "user_012": {
            "absolute_error_sum_rad": 123.33223596074822,
            "mae_deg": 5.537944042682189,
            "valid_angle_count": 1276
          },
          "user_013": {
            "absolute_error_sum_rad": 1448.3358329296752,
            "mae_deg": 6.959370223451451,
            "valid_angle_count": 11924
          },
          "user_014": {
            "absolute_error_sum_rad": 119.87174279941246,
            "mae_deg": 6.503925137579593,
            "valid_angle_count": 1056
          },
          "user_015": {
            "absolute_error_sum_rad": 191.21442674970604,
            "mae_deg": 6.303670675933019,
            "valid_angle_count": 1738
          },
          "user_016": {
            "absolute_error_sum_rad": 154.44942961004563,
            "mae_deg": 5.028011627760513,
            "valid_angle_count": 1760
          },
          "user_017": {
            "absolute_error_sum_rad": 15.216529740951955,
            "mae_deg": 4.953653028362743,
            "valid_angle_count": 176
          },
          "user_018": {
            "absolute_error_sum_rad": 65.32314372435212,
            "mae_deg": 5.670818848381615,
            "valid_angle_count": 660
          },
          "user_019": {
            "absolute_error_sum_rad": 171.07287444965914,
            "mae_deg": 4.546267947651616,
            "valid_angle_count": 2156
          },
          "user_020": {
            "absolute_error_sum_rad": 165.34332244767575,
            "mae_deg": 5.316203449451462,
            "valid_angle_count": 1782
          },
          "user_021": {
            "absolute_error_sum_rad": 189.7925714586454,
            "mae_deg": 5.043744586046135,
            "valid_angle_count": 2156
          },
          "user_022": {
            "absolute_error_sum_rad": 96.1391761682462,
            "mae_deg": 8.346013697432141,
            "valid_angle_count": 660
          },
          "user_023": {
            "absolute_error_sum_rad": 160.54750226711622,
            "mae_deg": 4.977648426013407,
            "valid_angle_count": 1848
          },
          "user_024": {
            "absolute_error_sum_rad": 139.0500385540654,
            "mae_deg": 5.0296593120450295,
            "valid_angle_count": 1584
          },
          "user_025": {
            "absolute_error_sum_rad": 1099.9991617055457,
            "mae_deg": 7.308129572548276,
            "valid_angle_count": 8624
          },
          "user_026": {
            "absolute_error_sum_rad": 108.80145530111622,
            "mae_deg": 6.746606270168005,
            "valid_angle_count": 924
          },
          "user_027": {
            "absolute_error_sum_rad": 228.31362131150672,
            "mae_deg": 6.606771164897708,
            "valid_angle_count": 1980
          },
          "user_028": {
            "absolute_error_sum_rad": 150.4721741619287,
            "mae_deg": 4.453213075225202,
            "valid_angle_count": 1936
          },
          "user_029": {
            "absolute_error_sum_rad": 100.34705504972953,
            "mae_deg": 5.226784309924034,
            "valid_angle_count": 1100
          },
          "user_030": {
            "absolute_error_sum_rad": 81.74468812672421,
            "mae_deg": 4.257841479340427,
            "valid_angle_count": 1100
          },
          "user_031": {
            "absolute_error_sum_rad": 665.4023844380044,
            "mae_deg": 6.876758352496238,
            "valid_angle_count": 5544
          },
          "user_032": {
            "absolute_error_sum_rad": 54.666075258515775,
            "mae_deg": 6.779513841682867,
            "valid_angle_count": 462
          },
          "user_033": {
            "absolute_error_sum_rad": 57.06467086961493,
            "mae_deg": 4.953886060806164,
            "valid_angle_count": 660
          },
          "user_034": {
            "absolute_error_sum_rad": 89.74772037991715,
            "mae_deg": 5.312154544100721,
            "valid_angle_count": 968
          },
          "user_035": {
            "absolute_error_sum_rad": 84.76656787941465,
            "mae_deg": 4.799176465711135,
            "valid_angle_count": 1012
          },
          "user_036": {
            "absolute_error_sum_rad": 98.4869793264661,
            "mae_deg": 5.829430012808573,
            "valid_angle_count": 968
          },
          "user_037": {
            "absolute_error_sum_rad": 110.20513088189182,
            "mae_deg": 4.6292440470815635,
            "valid_angle_count": 1364
          },
          "user_038": {
            "absolute_error_sum_rad": 1089.7175529218657,
            "mae_deg": 6.661994947049385,
            "valid_angle_count": 9372
          },
          "user_039": {
            "absolute_error_sum_rad": 1111.9530934473112,
            "mae_deg": 7.541455879622036,
            "valid_angle_count": 8448
          },
          "user_040": {
            "absolute_error_sum_rad": 217.00383562716888,
            "mae_deg": 9.744046958924315,
            "valid_angle_count": 1276
          }
        },
        "invalid_angle_count": 2816,
        "mode": "center",
        "overall": {
          "absolute_error_sum_rad": 10370.073765443363,
          "mae_deg": 6.464178815431942,
          "valid_angle_count": 91916
        },
        "user_summary": {
          "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
          "mean_mae_deg": 6.332663682889197,
          "std_mae_deg": 2.023265587836175,
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
            "absolute_error_sum_rad": 9071.119234500591,
            "mae_deg": 6.287646353699806,
            "valid_angle_count": 82660
          },
          "wrist": {
            "absolute_error_sum_rad": 965.9734780334929,
            "mae_deg": 6.695645222948498,
            "valid_angle_count": 8266
          }
        },
        "by_hand": {
          "left": {
            "absolute_error_sum_rad": 5162.1782717504,
            "mae_deg": 6.485353419844654,
            "valid_angle_count": 45606
          },
          "right": {
            "absolute_error_sum_rad": 4874.914440783684,
            "mae_deg": 6.163107302389298,
            "valid_angle_count": 45320
          }
        },
        "by_hand_split": {
          "left": {
            "both": {
              "absolute_error_sum_rad": 447.66454230234376,
              "mae_deg": 7.473569030180653,
              "valid_angle_count": 3432
            },
            "gesture": {
              "absolute_error_sum_rad": 2426.444780683025,
              "mae_deg": 5.713671097923886,
              "valid_angle_count": 24332
            },
            "user": {
              "absolute_error_sum_rad": 2288.0689487650316,
              "mae_deg": 7.347645667479615,
              "valid_angle_count": 17842
            }
          },
          "right": {
            "both": {
              "absolute_error_sum_rad": 423.0800841942164,
              "mae_deg": 7.018153798601079,
              "valid_angle_count": 3454
            },
            "gesture": {
              "absolute_error_sum_rad": 2344.896105821077,
              "mae_deg": 5.58221083015663,
              "valid_angle_count": 24068
            },
            "user": {
              "absolute_error_sum_rad": 2106.9382507683904,
              "mae_deg": 6.782709824907577,
              "valid_angle_count": 17798
            }
          }
        },
        "by_split": {
          "both": {
            "absolute_error_sum_rad": 870.7446264965602,
            "mae_deg": 7.245133914020932,
            "valid_angle_count": 6886
          },
          "gesture": {
            "absolute_error_sum_rad": 4771.340886504102,
            "mae_deg": 5.648299492043259,
            "valid_angle_count": 48400
          },
          "user": {
            "absolute_error_sum_rad": 4395.007199533422,
            "mae_deg": 7.065526472022344,
            "valid_angle_count": 35640
          }
        },
        "by_user": {
          "user_000": {
            "absolute_error_sum_rad": 122.55383310094476,
            "mae_deg": 5.699527110255537,
            "valid_angle_count": 1232
          },
          "user_001": {
            "absolute_error_sum_rad": 158.67461123410612,
            "mae_deg": 5.437431542819019,
            "valid_angle_count": 1672
          },
          "user_002": {
            "absolute_error_sum_rad": 269.7971841605613,
            "mae_deg": 10.646170782998613,
            "valid_angle_count": 1452
          },
          "user_003": {
            "absolute_error_sum_rad": 7.149819253361784,
            "mae_deg": 18.620657613591256,
            "valid_angle_count": 22
          },
          "user_004": {
            "absolute_error_sum_rad": 87.13938933174359,
            "mae_deg": 4.631464970367549,
            "valid_angle_count": 1078
          },
          "user_005": {
            "absolute_error_sum_rad": 157.50872248054657,
            "mae_deg": 6.724728043690513,
            "valid_angle_count": 1342
          },
          "user_006": {
            "absolute_error_sum_rad": 130.84855284783407,
            "mae_deg": 4.7329986323077025,
            "valid_angle_count": 1584
          },
          "user_007": {
            "absolute_error_sum_rad": 162.73158623103518,
            "mae_deg": 5.886258260421424,
            "valid_angle_count": 1584
          },
          "user_008": {
            "absolute_error_sum_rad": 175.40573039988521,
            "mae_deg": 5.438316046711572,
            "valid_angle_count": 1848
          },
          "user_009": {
            "absolute_error_sum_rad": 211.72682430263376,
            "mae_deg": 6.487194354143756,
            "valid_angle_count": 1870
          },
          "user_010": {
            "absolute_error_sum_rad": 125.57464700042328,
            "mae_deg": 5.274851383408649,
            "valid_angle_count": 1364
          },
          "user_011": {
            "absolute_error_sum_rad": 240.3096915874048,
            "mae_deg": 7.111947884322706,
            "valid_angle_count": 1936
          },
          "user_012": {
            "absolute_error_sum_rad": 123.33223596074822,
            "mae_deg": 5.537944042682189,
            "valid_angle_count": 1276
          },
          "user_013": {
            "absolute_error_sum_rad": 1448.3358329296752,
            "mae_deg": 6.959370223451451,
            "valid_angle_count": 11924
          },
          "user_014": {
            "absolute_error_sum_rad": 119.87174279941246,
            "mae_deg": 6.503925137579593,
            "valid_angle_count": 1056
          },
          "user_015": {
            "absolute_error_sum_rad": 191.21442674970604,
            "mae_deg": 6.303670675933019,
            "valid_angle_count": 1738
          },
          "user_016": {
            "absolute_error_sum_rad": 154.44942961004563,
            "mae_deg": 5.028011627760513,
            "valid_angle_count": 1760
          },
          "user_017": {
            "absolute_error_sum_rad": 15.216529740951955,
            "mae_deg": 4.953653028362743,
            "valid_angle_count": 176
          },
          "user_018": {
            "absolute_error_sum_rad": 65.32314372435212,
            "mae_deg": 5.670818848381615,
            "valid_angle_count": 660
          },
          "user_019": {
            "absolute_error_sum_rad": 171.07287444965914,
            "mae_deg": 4.546267947651616,
            "valid_angle_count": 2156
          },
          "user_020": {
            "absolute_error_sum_rad": 165.34332244767575,
            "mae_deg": 5.316203449451462,
            "valid_angle_count": 1782
          },
          "user_021": {
            "absolute_error_sum_rad": 189.7925714586454,
            "mae_deg": 5.043744586046135,
            "valid_angle_count": 2156
          },
          "user_022": {
            "absolute_error_sum_rad": 96.1391761682462,
            "mae_deg": 8.346013697432141,
            "valid_angle_count": 660
          },
          "user_023": {
            "absolute_error_sum_rad": 160.54750226711622,
            "mae_deg": 4.977648426013407,
            "valid_angle_count": 1848
          },
          "user_024": {
            "absolute_error_sum_rad": 139.0500385540654,
            "mae_deg": 5.0296593120450295,
            "valid_angle_count": 1584
          },
          "user_025": {
            "absolute_error_sum_rad": 1078.3483833150785,
            "mae_deg": 7.238145643007207,
            "valid_angle_count": 8536
          },
          "user_026": {
            "absolute_error_sum_rad": 96.78217483952176,
            "mae_deg": 6.301375170911162,
            "valid_angle_count": 880
          },
          "user_027": {
            "absolute_error_sum_rad": 228.31362131150672,
            "mae_deg": 6.606771164897708,
            "valid_angle_count": 1980
          },
          "user_028": {
            "absolute_error_sum_rad": 150.4721741619287,
            "mae_deg": 4.453213075225202,
            "valid_angle_count": 1936
          },
          "user_029": {
            "absolute_error_sum_rad": 100.34705504972953,
            "mae_deg": 5.226784309924034,
            "valid_angle_count": 1100
          },
          "user_030": {
            "absolute_error_sum_rad": 81.74468812672421,
            "mae_deg": 4.257841479340427,
            "valid_angle_count": 1100
          },
          "user_031": {
            "absolute_error_sum_rad": 617.5593326104918,
            "mae_deg": 6.5915691798615885,
            "valid_angle_count": 5368
          },
          "user_032": {
            "absolute_error_sum_rad": 41.72134114615619,
            "mae_deg": 5.71879608445037,
            "valid_angle_count": 418
          },
          "user_033": {
            "absolute_error_sum_rad": 57.06467086961493,
            "mae_deg": 4.953886060806164,
            "valid_angle_count": 660
          },
          "user_034": {
            "absolute_error_sum_rad": 89.74772037991715,
            "mae_deg": 5.312154544100721,
            "valid_angle_count": 968
          },
          "user_035": {
            "absolute_error_sum_rad": 84.76656787941465,
            "mae_deg": 4.799176465711135,
            "valid_angle_count": 1012
          },
          "user_036": {
            "absolute_error_sum_rad": 98.4869793264661,
            "mae_deg": 5.829430012808573,
            "valid_angle_count": 968
          },
          "user_037": {
            "absolute_error_sum_rad": 110.20513088189182,
            "mae_deg": 4.6292440470815635,
            "valid_angle_count": 1364
          },
          "user_038": {
            "absolute_error_sum_rad": 1089.7175529218657,
            "mae_deg": 6.661994947049385,
            "valid_angle_count": 9372
          },
          "user_039": {
            "absolute_error_sum_rad": 1111.9530934473112,
            "mae_deg": 7.541455879622036,
            "valid_angle_count": 8448
          },
          "user_040": {
            "absolute_error_sum_rad": 110.75280747568468,
            "mae_deg": 6.009155717406898,
            "valid_angle_count": 1056
          }
        },
        "invalid_angle_count": 0,
        "mode": "center",
        "overall": {
          "absolute_error_sum_rad": 10037.092712534084,
          "mae_deg": 6.324737159995141,
          "valid_angle_count": 90926
        },
        "user_summary": {
          "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
          "mean_mae_deg": 6.1716944246349605,
          "std_mae_deg": 2.3015894965943495,
          "user_count": 41
        }
      }
    },
    "unique_rule": "first original repetition only; duplicated consumptions remain in native and consumed metrics",
    "unique_samples": 4306
  },
  "training": {
    "completed_epochs": 124,
    "fit_result_sha256": "c37fec4cf32569e7aa6eec0bcb69379e3bf38d6c1d8f767a0f65f50b4afd2067",
    "full_training_terminated": true,
    "optimizer_executions": 69688,
    "scope": "full",
    "selected_best": {
      "checkpoint_sha256": "041050c2fc5c67050203744b08a25c19f449cc08ea5fdfd47d37fccb7e478b7b",
      "file": "vit-small-epoch=093-val_mae=0.1012.ckpt",
      "score": 0.10121652483940125,
      "validated": true
    }
  },
  "validation_profile_sha256": "290c7f9e7c7337a34a029ae525624a1cdc068d89fa8f709fea43f46458d1d0fd",
  "variant": "vit_small"
}
```

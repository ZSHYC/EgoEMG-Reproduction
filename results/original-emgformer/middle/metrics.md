# 原生 EMGFormer 中心测试报告

规模：middle；验证最优检查点：best-val_mae=val_mae=0.2578.ckpt。

| 唯一手样本 | 重复消费 | pooled MAE (°) | 有效角度数 |
| ---: | ---: | ---: | ---: |
| 2792 | 0 | 16.441148 | 59576 |

原生按样本数加权批量 MAE：0.2868970633 rad。

保留原生中心标签与生成掩码；总体指标按有效角度数汇总，分别列出左右手、条件、匿名用户、22关节和指组。
采用可用旧版数据与独立训练留出验证；未证明与PDF原表人口等价，未取得全论文完成资格。

```json
{
  "actual_replay_seconds": 34.131586849,
  "actual_test_pass_seconds": 25.46133282,
  "fit_result_sha256": "9010f796b2f195a3907a14fae0f5d671f9f19c774e3ca4bb89bcbdfcba23c051",
  "format": "original_emgformer_completed_point_report_v1",
  "full_paper_complete": false,
  "independent_test_sha256": "3bd5427c982d61f17a9f7019aea674851e77d66dcd11f8a86b084b2d3ac02ce2",
  "limitations": [
    "Legacy prefiltered temporal validation: official filtered EMG was processed over whole episodes; disjoint row footprints do not remove upstream filter dependence.",
    "Generated labels are the public release labels; public label-generation code differs from the paper description. No labels are regenerated here.",
    "Validation is a deterministic contiguous temporal holdout within official training runs, not a new unseen-subject benchmark.",
    "Short available training runs with fewer than two full windows contribute training only; validation fraction is approximate."
  ],
  "native_test_metrics": {
    "test_mae": 0.28689709305763245
  },
  "native_test_sha256": "88246f7c179e6a788b3369a1be626f04c24ba2842a309d1be8dc7c54ba9ef8ba",
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
  "sha256": "e7dede32c6dcce223748544eabef6a8a4105ba4ae1499c00423e04fd863ae8aa",
  "size": "middle",
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
          "absolute_error_sum_rad": 15763.629410637572,
          "mae_deg": 16.676318963032312,
          "valid_angle_count": 54160
        },
        "wrist": {
          "absolute_error_sum_rad": 1331.8328822150943,
          "mae_deg": 14.089439285020122,
          "valid_angle_count": 5416
        }
      },
      "by_finger": {
        "index": {
          "absolute_error_sum_rad": 2844.9521846542557,
          "mae_deg": 15.048352390806139,
          "valid_angle_count": 10832
        },
        "middle": {
          "absolute_error_sum_rad": 3301.174849208226,
          "mae_deg": 17.461538616540555,
          "valid_angle_count": 10832
        },
        "pinky": {
          "absolute_error_sum_rad": 3635.5949293464582,
          "mae_deg": 19.230451022037922,
          "valid_angle_count": 10832
        },
        "ring": {
          "absolute_error_sum_rad": 3336.056841428028,
          "mae_deg": 17.646046642316296,
          "valid_angle_count": 10832
        },
        "thumb": {
          "absolute_error_sum_rad": 2645.850606000604,
          "mae_deg": 13.995206143460658,
          "valid_angle_count": 10832
        }
      },
      "by_hand": {
        "left": {
          "absolute_error_sum_rad": 8346.629330261727,
          "mae_deg": 16.066204185449905,
          "valid_angle_count": 29766
        },
        "right": {
          "absolute_error_sum_rad": 8748.83296259094,
          "mae_deg": 16.815538558248818,
          "valid_angle_count": 29810
        }
      },
      "by_hand_split": {
        "left": {
          "both": {
            "absolute_error_sum_rad": 688.9222707029985,
            "mae_deg": 17.941972056295917,
            "valid_angle_count": 2200
          },
          "gesture": {
            "absolute_error_sum_rad": 4238.281110660886,
            "mae_deg": 15.203832959609805,
            "valid_angle_count": 15972
          },
          "user": {
            "absolute_error_sum_rad": 3419.4259488978423,
            "mae_deg": 16.89828145845809,
            "valid_angle_count": 11594
          }
        },
        "right": {
          "both": {
            "absolute_error_sum_rad": 736.6806300292374,
            "mae_deg": 18.995810508422032,
            "valid_angle_count": 2222
          },
          "gesture": {
            "absolute_error_sum_rad": 4336.81608672611,
            "mae_deg": 15.578762275476347,
            "valid_angle_count": 15950
          },
          "user": {
            "absolute_error_sum_rad": 3675.3362458355914,
            "mae_deg": 18.094282108423766,
            "valid_angle_count": 11638
          }
        }
      },
      "by_joint": [
        {
          "absolute_error_sum_rad": 557.5911568185402,
          "index": 0,
          "mae_deg": 11.797496299674885,
          "name": "THUMB_CMC_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 764.4843595642451,
          "index": 1,
          "mae_deg": 16.174936228505512,
          "name": "THUMB_CMC_AA",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 675.1908795230556,
          "index": 2,
          "mae_deg": 14.285667563662159,
          "name": "THUMB_MCP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 648.584210094763,
          "index": 3,
          "mae_deg": 13.722724482000075,
          "name": "THUMB_IP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 374.2610989592504,
          "index": 4,
          "mae_deg": 7.918604655204244,
          "name": "INDEX_MCP_AA",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 832.2473193228434,
          "index": 5,
          "mae_deg": 17.608662816940715,
          "name": "INDEX_MCP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1179.6678346601548,
          "index": 6,
          "mae_deg": 24.959375241271594,
          "name": "INDEX_PIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 458.77593171200715,
          "index": 7,
          "mae_deg": 9.706766849808004,
          "name": "INDEX_DIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 279.1497523891885,
          "index": 8,
          "mae_deg": 5.906241751854679,
          "name": "MIDDLE_MCP_AA",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1075.5571521313977,
          "index": 9,
          "mae_deg": 22.756604668478325,
          "name": "MIDDLE_MCP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1337.8474855962559,
          "index": 10,
          "mae_deg": 28.306135360729193,
          "name": "MIDDLE_PIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 608.620459091384,
          "index": 11,
          "mae_deg": 12.877172685100028,
          "name": "MIDDLE_DIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 271.77424053305003,
          "index": 12,
          "mae_deg": 5.750190902111165,
          "name": "RING_MCP_AA",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1097.5478586856043,
          "index": 13,
          "mae_deg": 23.221883351663966,
          "name": "RING_MCP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1509.8952212369186,
          "index": 14,
          "mae_deg": 31.9463159836954,
          "name": "RING_PIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 456.8395209724549,
          "index": 15,
          "mae_deg": 9.665796331794654,
          "name": "RING_DIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 569.5792171925787,
          "index": 16,
          "mae_deg": 12.051139307053182,
          "name": "PINKY_MCP_AA",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1385.523903907524,
          "index": 17,
          "mae_deg": 29.31487153190198,
          "name": "PINKY_MCP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 1190.55156668555,
          "index": 18,
          "mae_deg": 25.189652903903248,
          "name": "PINKY_PIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 489.9402415608056,
          "index": 19,
          "mae_deg": 10.366140345293282,
          "name": "PINKY_DIP_FE",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 811.0265411608852,
          "index": 20,
          "mae_deg": 17.15967425465726,
          "name": "wrist_pitch",
          "valid_angle_count": 2708
        },
        {
          "absolute_error_sum_rad": 520.806341054209,
          "index": 21,
          "mae_deg": 11.019204315382982,
          "name": "wrist_yaw",
          "valid_angle_count": 2708
        }
      ],
      "by_pd_group": {
        "distal": {
          "absolute_error_sum_rad": 2662.7603634314146,
          "mae_deg": 11.267720138799207,
          "valid_angle_count": 13540
        },
        "mid": {
          "absolute_error_sum_rad": 5893.152987701935,
          "mae_deg": 24.937429410652317,
          "valid_angle_count": 13540
        },
        "proximal": {
          "absolute_error_sum_rad": 7207.716059504222,
          "mae_deg": 15.250063151338864,
          "valid_angle_count": 27080
        }
      },
      "by_split": {
        "both": {
          "absolute_error_sum_rad": 1425.6029007322359,
          "mae_deg": 18.471512771045855,
          "valid_angle_count": 4422
        },
        "gesture": {
          "absolute_error_sum_rad": 8575.097197386996,
          "mae_deg": 15.391168420673377,
          "valid_angle_count": 31922
        },
        "user": {
          "absolute_error_sum_rad": 7094.762194733434,
          "mae_deg": 17.497414359814,
          "valid_angle_count": 23232
        }
      },
      "by_split_user_summary": {
        "both": {
          "aggregation": "unweighted user means; ddof=0",
          "mean_mae_deg": 18.23438584366439,
          "std_mae_deg": 1.0453681673806023,
          "user_count": 5
        },
        "gesture": {
          "aggregation": "unweighted user means; ddof=0",
          "mean_mae_deg": 15.403129309023093,
          "std_mae_deg": 1.7622784614118805,
          "user_count": 36
        },
        "user": {
          "aggregation": "unweighted user means; ddof=0",
          "mean_mae_deg": 17.333991714752823,
          "std_mae_deg": 0.6738534809341832,
          "user_count": 5
        }
      },
      "by_user": {
        "user_000": {
          "absolute_error_sum_rad": 228.73525807631813,
          "mae_deg": 15.676513054555867,
          "valid_angle_count": 836
        },
        "user_001": {
          "absolute_error_sum_rad": 308.810257228557,
          "mae_deg": 16.085022190496037,
          "valid_angle_count": 1100
        },
        "user_002": {
          "absolute_error_sum_rad": 257.5757200922235,
          "mae_deg": 14.583005599137792,
          "valid_angle_count": 1012
        },
        "user_003": {
          "absolute_error_sum_rad": 210.34993161354214,
          "mae_deg": 14.416463280288824,
          "valid_angle_count": 836
        },
        "user_004": {
          "absolute_error_sum_rad": 305.8368526719496,
          "mae_deg": 18.523425874912334,
          "valid_angle_count": 946
        },
        "user_005": {
          "absolute_error_sum_rad": 277.61260091906297,
          "mae_deg": 14.176497658032039,
          "valid_angle_count": 1122
        },
        "user_006": {
          "absolute_error_sum_rad": 237.33253524254542,
          "mae_deg": 13.436909694207236,
          "valid_angle_count": 1012
        },
        "user_007": {
          "absolute_error_sum_rad": 336.63601131748874,
          "mae_deg": 15.940349322818447,
          "valid_angle_count": 1210
        },
        "user_008": {
          "absolute_error_sum_rad": 351.3859924044955,
          "mae_deg": 16.946914431644437,
          "valid_angle_count": 1188
        },
        "user_009": {
          "absolute_error_sum_rad": 222.3827052996494,
          "mae_deg": 14.479080057240312,
          "valid_angle_count": 880
        },
        "user_010": {
          "absolute_error_sum_rad": 313.7934431396425,
          "mae_deg": 14.59337656719147,
          "valid_angle_count": 1232
        },
        "user_011": {
          "absolute_error_sum_rad": 248.1956852858275,
          "mae_deg": 17.01024552659706,
          "valid_angle_count": 836
        },
        "user_012": {
          "absolute_error_sum_rad": 2457.4673034050065,
          "mae_deg": 18.39113176286602,
          "valid_angle_count": 7656
        },
        "user_013": {
          "absolute_error_sum_rad": 165.68197733932175,
          "mae_deg": 17.259778259864145,
          "valid_angle_count": 550
        },
        "user_014": {
          "absolute_error_sum_rad": 368.19784158341645,
          "mae_deg": 17.75772924962647,
          "valid_angle_count": 1188
        },
        "user_015": {
          "absolute_error_sum_rad": 355.9793575670046,
          "mae_deg": 17.168446786504873,
          "valid_angle_count": 1188
        },
        "user_016": {
          "absolute_error_sum_rad": 40.984597681090236,
          "mae_deg": 17.78973084975857,
          "valid_angle_count": 132
        },
        "user_017": {
          "absolute_error_sum_rad": 107.93524434510618,
          "mae_deg": 14.05507718565425,
          "valid_angle_count": 440
        },
        "user_018": {
          "absolute_error_sum_rad": 344.30459127615904,
          "mae_deg": 14.462756559458018,
          "valid_angle_count": 1364
        },
        "user_019": {
          "absolute_error_sum_rad": 272.4336030703853,
          "mae_deg": 12.900244341715194,
          "valid_angle_count": 1210
        },
        "user_020": {
          "absolute_error_sum_rad": 364.44510485685896,
          "mae_deg": 14.830373844105653,
          "valid_angle_count": 1408
        },
        "user_021": {
          "absolute_error_sum_rad": 141.83739261628943,
          "mae_deg": 18.469736304666842,
          "valid_angle_count": 440
        },
        "user_022": {
          "absolute_error_sum_rad": 288.7334756159835,
          "mae_deg": 13.42792983517802,
          "valid_angle_count": 1232
        },
        "user_023": {
          "absolute_error_sum_rad": 283.3459324783762,
          "mae_deg": 16.04202181147209,
          "valid_angle_count": 1012
        },
        "user_024": {
          "absolute_error_sum_rad": 1698.9141549180786,
          "mae_deg": 17.557830236623523,
          "valid_angle_count": 5544
        },
        "user_025": {
          "absolute_error_sum_rad": 170.3753782210406,
          "mae_deg": 15.847061858783686,
          "valid_angle_count": 616
        },
        "user_026": {
          "absolute_error_sum_rad": 371.3687343407073,
          "mae_deg": 17.270991169511046,
          "valid_angle_count": 1232
        },
        "user_027": {
          "absolute_error_sum_rad": 377.1766194589436,
          "mae_deg": 16.371688201522264,
          "valid_angle_count": 1320
        },
        "user_028": {
          "absolute_error_sum_rad": 176.8006562330993,
          "mae_deg": 13.542689060561392,
          "valid_angle_count": 748
        },
        "user_029": {
          "absolute_error_sum_rad": 183.2852248735726,
          "mae_deg": 14.916860557332281,
          "valid_angle_count": 704
        },
        "user_030": {
          "absolute_error_sum_rad": 939.9899196838087,
          "mae_deg": 15.300413404182803,
          "valid_angle_count": 3520
        },
        "user_031": {
          "absolute_error_sum_rad": 66.58286103978753,
          "mae_deg": 13.338870368831698,
          "valid_angle_count": 286
        },
        "user_032": {
          "absolute_error_sum_rad": 100.46714193373919,
          "mae_deg": 14.536220233699666,
          "valid_angle_count": 396
        },
        "user_033": {
          "absolute_error_sum_rad": 189.8539477775339,
          "mae_deg": 17.658814823944525,
          "valid_angle_count": 616
        },
        "user_034": {
          "absolute_error_sum_rad": 168.23685029475018,
          "mae_deg": 14.604941637065824,
          "valid_angle_count": 660
        },
        "user_035": {
          "absolute_error_sum_rad": 167.93031761609018,
          "mae_deg": 14.578330987414224,
          "valid_angle_count": 660
        },
        "user_036": {
          "absolute_error_sum_rad": 216.5036443436402,
          "mae_deg": 14.457744836937064,
          "valid_angle_count": 858
        },
        "user_037": {
          "absolute_error_sum_rad": 1829.1680182549608,
          "mae_deg": 17.19783516020971,
          "valid_angle_count": 6094
        },
        "user_038": {
          "absolute_error_sum_rad": 1715.6443391562716,
          "mae_deg": 17.87257814166647,
          "valid_angle_count": 5500
        },
        "user_039": {
          "absolute_error_sum_rad": 233.17106958034128,
          "mae_deg": 16.868331049879814,
          "valid_angle_count": 792
        }
      },
      "invalid_angle_count": 1848,
      "mode": "center",
      "overall": {
        "absolute_error_sum_rad": 17095.462292852666,
        "mae_deg": 16.441148083213022,
        "valid_angle_count": 59576
      },
      "user_summary": {
        "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
        "mean_mae_deg": 15.758599044403951,
        "std_mae_deg": 1.6239996806344918,
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
    "native_batch_weighted_mae_rad": 0.28689706325531006,
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
      "protocol_sha256": "c2a09e84f03aad228acbf9cd1e06451d98e9b9d7a460a541fc57827ed27f94f3",
      "save_last": true,
      "save_top_k": 3,
      "scheduler_T_max": 200,
      "scheduler_eta_min": 1e-05,
      "scientific_eligible": false,
      "scope": "full",
      "sha256": "f058e920d8611b1f7dc18e847ba5c06b454af5e806a321df21c4e17d171ed308",
      "single_rank": true,
      "size": "middle",
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
      "checkpoint_sha256": "c9a62af344490dc530a89e4f6b861a5463cd8e9b8aa05fb5380d171776ee36ff",
      "file": "best-val_mae=val_mae=0.2578.ckpt",
      "score": 0.25775474309921265,
      "validated": true
    }
  },
  "validation_profile_sha256": "19d9e281bae88fb55826d87b527a9e644a2bda2cbf32e88c2f2a996ea6dc8c71",
  "vision_subset_reported": false
}
```

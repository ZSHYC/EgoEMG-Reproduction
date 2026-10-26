# Classic selected-best experiment

Variant: **vemg2pose**. This is one legacy-release experiment; the full paper remains incomplete.

Report SHA256: `598726fa45bc46a4c2bfbab34bc65a1fd77ae9d86284c51525ee1213a84efa69`. Completion requires its bound `completion.json`.

Selection: original val_loss/min; epoch 11 (zero-based); 62 completed epochs, 10535 optimizer executions, 5 overflow skips.

Angular sums are radians; MAE is degrees. Aggregate sums/counts before division. Native metrics retain their own original aggregation.

| Mode / group | Error sum (rad) | Valid angles | MAE (degrees) |
| --- | ---: | ---: | ---: |
| trajectory / overall | 158274782.376 | 551176956 | 16.452932102 |
| trajectory / by_hand / left | 78863408.5221 | 276033252 | 16.369551254 |
| trajectory / by_hand / right | 79411373.8539 | 275143704 | 16.536582524 |
| trajectory / by_split / both | 12390297.5739 | 41229254 | 17.218641839 |
| trajectory / by_split / gesture | 82906873.7861 | 294394518 | 16.135538096 |
| trajectory / by_split / user | 62977611.016 | 215553184 | 16.739958316 |
| trajectory / by_component / fingers | 145243912.985 | 501069960 | 16.608186238 |
| trajectory / by_component / wrist | 13030869.3911 | 50106996 | 14.900390746 |
| trajectory / by_finger / index | 25778098.7175 | 100213992 | 14.738223984 |
| trajectory / by_finger / middle | 30818190.1231 | 100213992 | 17.619817263 |
| trajectory / by_finger / pinky | 33803316.9926 | 100213992 | 19.326516772 |
| trajectory / by_finger / ring | 30829247.0694 | 100213992 | 17.626138899 |
| trajectory / by_finger / thumb | 24015060.0823 | 100213992 | 13.730234272 |
| trajectory / by_pd_group / distal | 23854733.0979 | 125267490 | 10.910855865 |
| trajectory / by_pd_group / mid | 54005592.6996 | 125267490 | 24.701481061 |
| trajectory / by_pd_group / proximal | 67383587.1875 | 250534980 | 15.410204013 |
| trajectory / by_user / participant_001 | 2188473.87361 | 7392000 | 16.962975721 |
| trajectory / by_user / participant_002 | 2945950.48011 | 10296000 | 16.393796539 |
| trajectory / by_user / participant_003 | 2326327.6991 | 8690264 | 15.337711135 |
| trajectory / by_user / participant_004 | 40908.9570566 | 264000 | 8.878449180 |
| trajectory / by_user / participant_005 | 2061416.48558 | 8247360 | 14.321002653 |
| trajectory / by_user / participant_006 | 2604486.60786 | 8032244 | 18.578381138 |
| trajectory / by_user / participant_007 | 2660501.03098 | 9801220 | 15.552704711 |
| trajectory / by_user / participant_008 | 2471403.67101 | 9504000 | 14.899095099 |
| trajectory / by_user / participant_009 | 3136286.11409 | 11070180 | 16.232433229 |
| trajectory / by_user / participant_010 | 3276694.36632 | 11088484 | 16.931147481 |
| trajectory / by_user / participant_011 | 2211641.70173 | 8029362 | 15.781793784 |
| trajectory / by_user / participant_012 | 3204510.60173 | 11613448 | 15.809683126 |
| trajectory / by_user / participant_013 | 2295454.12579 | 7575920 | 17.360245815 |
| trajectory / by_user / participant_014 | 20628793.8398 | 71338366 | 16.568123013 |
| trajectory / by_user / participant_015 | 1981001.47609 | 6335076 | 17.916600178 |
| trajectory / by_user / participant_016 | 3110184.45371 | 10557052 | 16.879754188 |
| trajectory / by_user / participant_017 | 3232180.06415 | 10560000 | 17.536957983 |
| trajectory / by_user / participant_018 | 304886.330621 | 1056000 | 16.542329523 |
| trajectory / by_user / participant_019 | 1185157.02253 | 4030422 | 16.847986502 |
| trajectory / by_user / participant_020 | 3479627.39884 | 12936000 | 15.411871075 |
| trajectory / by_user / participant_021 | 2706616.76935 | 10782134 | 14.382840878 |
| trajectory / by_user / participant_022 | 3435141.53649 | 12936000 | 15.214835503 |
| trajectory / by_user / participant_023 | 1295100.10309 | 3899368 | 19.029691466 |
| trajectory / by_user / participant_024 | 2831257.30386 | 11088000 | 14.630149191 |
| trajectory / by_user / participant_025 | 2621852.57874 | 9504000 | 15.806090832 |
| trajectory / by_user / participant_026 | 16093061.6587 | 51703652 | 17.833643791 |
| trajectory / by_user / participant_027 | 1515711.05138 | 5544000 | 15.664474424 |
| trajectory / by_user / participant_028 | 3726970.30961 | 11880000 | 17.974719622 |
| trajectory / by_user / participant_029 | 3518230.21676 | 11706684 | 17.219200824 |
| trajectory / by_user / participant_030 | 1713707.79843 | 6600000 | 14.877003662 |
| trajectory / by_user / participant_031 | 1662760.77358 | 6600000 | 14.434723434 |
| trajectory / by_user / participant_032 | 9042266.771 | 33264000 | 15.574907504 |
| trajectory / by_user / participant_033 | 694965.111838 | 2683098 | 14.840519361 |
| trajectory / by_user / participant_034 | 1069444.07685 | 3960000 | 15.473391926 |
| trajectory / by_user / participant_035 | 1651524.54237 | 5808000 | 16.292249663 |
| trajectory / by_user / participant_036 | 1605086.02074 | 6072000 | 15.145694128 |
| trajectory / by_user / participant_037 | 1833846.91038 | 5766992 | 18.219496097 |
| trajectory / by_user / participant_038 | 2339354.39702 | 8248966 | 16.248719383 |
| trajectory / by_user / participant_039 | 16170397.914 | 56388420 | 16.430599643 |
| trajectory / by_user / participant_040 | 14973876.4351 | 50688000 | 16.925898096 |
| trajectory / by_user / participant_041 | 2427723.79597 | 7636244 | 18.215542528 |
| trajectory / THUMB_CMC_FE | 5407347.32082 | 25053498 | 12.366264377 |
| trajectory / THUMB_CMC_AA | 6684130.97947 | 25053498 | 15.286188573 |
| trajectory / THUMB_MCP_FE | 6112314.65319 | 25053498 | 13.978480477 |
| trajectory / THUMB_IP_FE | 5811267.12883 | 25053498 | 13.290003660 |
| trajectory / INDEX_MCP_AA | 3484436.01639 | 25053498 | 7.968686757 |
| trajectory / INDEX_MCP_FE | 7701025.76641 | 25053498 | 17.611763209 |
| trajectory / INDEX_PIP_FE | 10511571.9598 | 25053498 | 24.039306182 |
| trajectory / INDEX_DIP_FE | 4081064.97492 | 25053498 | 9.333139787 |
| trajectory / MIDDLE_MCP_AA | 2663575.298 | 25053498 | 6.091429747 |
| trajectory / MIDDLE_MCP_FE | 10226803.9676 | 25053498 | 23.388059633 |
| trajectory / MIDDLE_PIP_FE | 12433076.681 | 25053498 | 28.433667035 |
| trajectory / MIDDLE_DIP_FE | 5494734.17645 | 25053498 | 12.566112639 |
| trajectory / RING_MCP_AA | 2524724.03923 | 25053498 | 5.773885622 |
| trajectory / RING_MCP_FE | 10408906.0353 | 25053498 | 23.804515648 |
| trajectory / RING_PIP_FE | 13767826.9107 | 25053498 | 31.486157145 |
| trajectory / RING_DIP_FE | 4127790.08419 | 25053498 | 9.439997183 |
| trajectory / PINKY_MCP_AA | 5429890.40821 | 25053498 | 12.417819005 |
| trajectory / PINKY_MCP_FE | 12852747.356 | 25053498 | 29.393427562 |
| trajectory / PINKY_PIP_FE | 11180802.4949 | 25053498 | 25.569794466 |
| trajectory / PINKY_DIP_FE | 4339876.73349 | 25053498 | 9.925026056 |
| trajectory / wrist_pitch | 7863246.62042 | 25053498 | 17.982752134 |
| trajectory / wrist_yaw | 5167622.77063 | 25053498 | 11.818029358 |
| center / overall | 26225.5901941 | 91916 | 16.347704789 |
| center / by_hand / left | 13059.8803502 | 46112 | 16.227360015 |
| center / by_hand / right | 13165.7098439 | 45804 | 16.468858797 |
| center / by_split / both | 2056.49658519 | 6886 | 17.111323688 |
| center / by_split / gesture | 13810.6646153 | 49104 | 16.114630065 |
| center / by_split / user | 10358.4289936 | 35926 | 16.519909361 |
| center / by_component / fingers | 24048.045626 | 83560 | 16.489367160 |
| center / by_component / wrist | 2177.54456803 | 8356 | 14.931081073 |
| center / by_finger / index | 4260.14669082 | 16712 | 14.605578356 |
| center / by_finger / middle | 5098.51403714 | 16712 | 17.479854961 |
| center / by_finger / pinky | 5592.84510056 | 16712 | 19.174630190 |
| center / by_finger / ring | 5114.1343023 | 16712 | 17.533407814 |
| center / by_finger / thumb | 3982.40549523 | 16712 | 13.653364480 |
| center / by_pd_group / distal | 3949.02693018 | 20890 | 10.831142953 |
| center / by_pd_group / mid | 8955.89314953 | 20890 | 24.563661045 |
| center / by_pd_group / proximal | 11143.1255463 | 41780 | 15.281332321 |
| center / by_user / participant_001 | 380.011351545 | 1232 | 17.672927444 |
| center / by_user / participant_002 | 521.43953444 | 1716 | 17.410422258 |
| center / by_user / participant_003 | 390.365994861 | 1452 | 15.403804388 |
| center / by_user / participant_004 | 9.49261517578 | 44 | 12.361063321 |
| center / by_user / participant_005 | 351.948352348 | 1386 | 14.549174023 |
| center / by_user / participant_006 | 427.466096361 | 1342 | 18.250374968 |
| center / by_user / participant_007 | 451.372619095 | 1628 | 15.885593404 |
| center / by_user / participant_008 | 412.691411673 | 1584 | 14.927699577 |
| center / by_user / participant_009 | 516.425064841 | 1848 | 16.011351001 |
| center / by_user / participant_010 | 557.994790222 | 1870 | 17.096655866 |
| center / by_user / participant_011 | 361.748012922 | 1364 | 15.195479756 |
| center / by_user / participant_012 | 555.381526587 | 1936 | 16.436475978 |
| center / by_user / participant_013 | 380.57892024 | 1276 | 17.089001490 |
| center / by_user / participant_014 | 3402.37935222 | 11924 | 16.348706574 |
| center / by_user / participant_015 | 314.551563748 | 1056 | 17.066739623 |
| center / by_user / participant_016 | 495.826352804 | 1738 | 16.345660177 |
| center / by_user / participant_017 | 539.887354075 | 1760 | 17.575719773 |
| center / by_user / participant_018 | 46.1795504047 | 176 | 15.033484875 |
| center / by_user / participant_019 | 194.481595438 | 660 | 16.883294866 |
| center / by_user / participant_020 | 561.062112214 | 2156 | 14.910246324 |
| center / by_user / participant_021 | 470.972943762 | 1782 | 15.142964053 |
| center / by_user / participant_022 | 561.758279735 | 2156 | 14.928747002 |
| center / by_user / participant_023 | 222.389955893 | 660 | 19.306069513 |
| center / by_user / participant_024 | 459.74131862 | 1848 | 14.253916247 |
| center / by_user / participant_025 | 436.897878206 | 1584 | 15.803285669 |
| center / by_user / participant_026 | 2695.61941373 | 8624 | 17.909046334 |
| center / by_user / participant_027 | 251.545829246 | 924 | 15.597959275 |
| center / by_user / participant_028 | 609.65426365 | 1980 | 17.641725389 |
| center / by_user / participant_029 | 594.571467767 | 1936 | 17.596299443 |
| center / by_user / participant_030 | 284.262363607 | 1100 | 14.806394281 |
| center / by_user / participant_031 | 277.265421193 | 1100 | 14.441944036 |
| center / by_user / participant_032 | 1474.99218603 | 5544 | 15.243655677 |
| center / by_user / participant_033 | 115.027467065 | 462 | 14.265342837 |
| center / by_user / participant_034 | 177.853968136 | 660 | 15.439820824 |
| center / by_user / participant_035 | 277.840225103 | 968 | 16.445322601 |
| center / by_user / participant_036 | 258.906178645 | 1012 | 14.658331350 |
| center / by_user / participant_037 | 298.095857948 | 968 | 17.644250569 |
| center / by_user / participant_038 | 391.185553114 | 1364 | 16.432024340 |
| center / by_user / participant_039 | 2659.20913653 | 9372 | 16.257091375 |
| center / by_user / participant_040 | 2423.59568634 | 8448 | 16.437240066 |
| center / by_user / participant_041 | 412.920628534 | 1276 | 18.541229850 |
| center / THUMB_CMC_FE | 892.961698161 | 4178 | 12.245796211 |
| center / THUMB_CMC_AA | 1109.26831165 | 4178 | 15.212157158 |
| center / THUMB_MCP_FE | 1013.80715786 | 4178 | 13.903032883 |
| center / THUMB_IP_FE | 966.368327563 | 4178 | 13.252471667 |
| center / INDEX_MCP_AA | 583.483749081 | 4178 | 8.001712838 |
| center / INDEX_MCP_FE | 1264.58244699 | 4178 | 17.342086419 |
| center / INDEX_PIP_FE | 1739.96109999 | 4178 | 23.861279930 |
| center / INDEX_DIP_FE | 672.11939476 | 4178 | 9.217234239 |
| center / MIDDLE_MCP_AA | 442.923559322 | 4178 | 6.074114551 |
| center / MIDDLE_MCP_FE | 1684.75570483 | 4178 | 23.104210483 |
| center / MIDDLE_PIP_FE | 2058.43193599 | 4178 | 28.228688929 |
| center / MIDDLE_DIP_FE | 912.402836998 | 4178 | 12.512405882 |
| center / RING_MCP_AA | 419.360861042 | 4178 | 5.750983109 |
| center / RING_MCP_FE | 1717.18469498 | 4178 | 23.548931466 |
| center / RING_PIP_FE | 2298.3997971 | 4178 | 31.519532793 |
| center / RING_DIP_FE | 679.188949176 | 4178 | 9.314183887 |
| center / PINKY_MCP_AA | 905.297615008 | 4178 | 12.414967100 |
| center / PINKY_MCP_FE | 2123.30690528 | 4178 | 29.118363878 |
| center / PINKY_PIP_FE | 1845.29315859 | 4178 | 25.305770692 |
| center / PINKY_DIP_FE | 718.947421682 | 4178 | 9.859419089 |
| center / wrist_pitch | 1307.32948999 | 4178 | 17.928305938 |
| center / wrist_yaw | 870.215078045 | 4178 | 11.933856209 |

Native original metrics (unchanged units; FK batch dependent):

| Metric | Value |
| --- | ---: |
| test_derivatives/acc | 0.127968519926 |
| test_derivatives/jerk | 0.254290103912 |
| test_derivatives/vel | 0.171465069056 |
| test_landmark/all | 27.6747341156 |
| test_landmark/fingertip | 46.7924575806 |
| test_loss | 1.22301697731 |
| test_mae | 0.287167847157 |
| test_mae_per_finger/index | 0.257244586945 |
| test_mae_per_finger/middle | 0.307753711939 |
| test_mae_per_finger/pinky | 0.337312877178 |
| test_mae_per_finger/ring | 0.307703733444 |
| test_mae_per_finger/thumb | 0.239509940147 |
| test_mae_per_pd/distal | 0.190563991666 |
| test_mae_per_pd/mid | 0.430835515261 |
| test_mae_per_pd/proximal | 0.269110172987 |
| test_mae_wrist/deviation | 0.206034839153 |
| test_mae_wrist/flexion | 0.313558012247 |

Limitations:

- Released legacy data/labels are not proven equivalent to PDF/current full release.
- Single rank and workers0 do not reproduce the original six-rank global batch or NeuroPose BatchNorm.
- Generated 2D label validity applies to all22 angles; raw marker/label regeneration is separate unfinished work.
- One seed42; no multi-seed or universal deterministic training trajectory claim.
- Exact prediction replay is observed under the frozen validation profile and this test pair only.
- Original FK remains batch dependent; native canonical batch is preserved.
- Subject overlap is counted from actual frozen windows; released condition names do not establish unseen-subject independence.

Complete public aggregate evidence (unrounded JSON; same content as metrics.json):

```json
{
  "completion_requires_owned_capsule": true,
  "data": {
    "hand_samples": {
      "test": 4306,
      "train": 101714,
      "val": 1576
    },
    "participant_labels": "sorted unique full-source subject keys; mapping private; natural-person uniqueness not proven",
    "source_subject_count": 41,
    "test_subject_count": 41,
    "train_test_subject_overlap": {
      "both": 1,
      "gesture": 36,
      "user": 1
    }
  },
  "evidence": {
    "aggregate_test_fingerprints": {
      "batch_sizes": [
        600,
        600,
        600,
        600,
        600,
        600,
        600,
        106
      ],
      "coverage_sha256": "346decce0c7554f81a971400b5acd21992db612bbcf4f885914d629a0bf8d0ff",
      "emg_sha256": "bcd46158f97d2860d0e5717a9eba6ec2526705b13497cc52591ec6fa25ba18f0",
      "mask_sha256": "9d303f1e73eaa13ea0e3f8ec5a732fdf3854212264cb18443b991ec83f33c292",
      "metadata_sha256": "5677338626c41f9c68d8d16e2b43aaacabc8db2bd5508f38019792dccef27d6a",
      "prediction_sha256": "1fa0457a7da16364086014e546a36db9666ca0690c91d1e382f2dd2a77b1a6fc",
      "sample_sha256": "8fac57ddec7f417c8aa64d161e8dd3dfce4280ff7f205fd24225b7e0648c087b",
      "samples": 4306,
      "target_sha256": "aa6486cbb310c6d4f95284d55bfaed652ee92089e23ae7c320dff32cb4b25203"
    },
    "cohort_bindings": {
      "cohort_receipt_sha256": "67de2924a9a96c04d394e6bf530ec81fbda11876af7afc72683bebd5f23c12e6",
      "data_sha256": "1306202563fdbe62342506005055ed82cb5ccd3fcd135663ab727541d6a0625b",
      "normalization_sha256": "7ff05f81f732536dbf2abf28b74dc978f8f72dedd65cfb15423d5ae2bdad743e",
      "numeric_qualification_artifact_sha256": "a31ea805fd7bc6493fa2931deefe397777a005cdf51ad711e93694ffe8836007",
      "partition_sha256": "c1f39851d24135d6f9cb08f6cfa15ea9aeaf0144a1b278ebc1453be6fa9dd5a4"
    },
    "native_launch_manifest_sha256": "8f7c5ff6f3cb41b9a123e7069962a78c56470e4fca0b71df099efad6ae57edc2",
    "native_launch_result_sha256": "a1c831bffdea1c8a47f1eed517172a7f9e58a641c8a160f62fcc82ee2d113aae",
    "native_test_result_sha256": "4522ba6f45c914fa615120b55097a5724fe00b0f9db21ce176c4a3b931428867",
    "recompute_result_sha256": "6e2d521a4ef1564c6ab492c74490e39a2800422ba05d2096bc18f5b62c088224",
    "reference_result_sha256": "45af0472afe8b31103fd0d6fc2096f6a383cbd754e626ef1317e34429699e0c4",
    "test_prediction_identity_sha256": "199a277ad3e2f794b602c86a02672463718f04d8deea0e881870630903b4975f",
    "training_manifest_sha256": "a26f8bd26892fa3b996c66b6010b93c8d52dce5e29fa7e1201bfac6f12a62ed7",
    "training_result_sha256": "98926825780b60cb326084c87905aed1f673d5c2b8505c17171876ea57a67ff1"
  },
  "format": "classic_original_22d_completed_report_v1",
  "full_paper_complete": false,
  "limitations": [
    "Released legacy data/labels are not proven equivalent to PDF/current full release.",
    "Single rank and workers0 do not reproduce the original six-rank global batch or NeuroPose BatchNorm.",
    "Generated 2D label validity applies to all22 angles; raw marker/label regeneration is separate unfinished work.",
    "One seed42; no multi-seed or universal deterministic training trajectory claim.",
    "Exact prediction replay is observed under the frozen validation profile and this test pair only.",
    "Original FK remains batch dependent; native canonical batch is preserved.",
    "Subject overlap is counted from actual frozen windows; released condition names do not establish unseen-subject independence."
  ],
  "native_metrics": {
    "aggregation": "unchanged original batch-weighted Trainer.test",
    "angular_units": "radians",
    "derivative_units": "original upstream sample-rate scaling; not reinterpreted",
    "landmark_units": "original upstream coordinate units; not converted",
    "values": {
      "test_derivatives/acc": 0.12796851992607117,
      "test_derivatives/jerk": 0.2542901039123535,
      "test_derivatives/vel": 0.17146506905555725,
      "test_landmark/all": 27.674734115600586,
      "test_landmark/fingertip": 46.792457580566406,
      "test_loss": 1.2230169773101807,
      "test_mae": 0.28716784715652466,
      "test_mae_per_finger/index": 0.2572445869445801,
      "test_mae_per_finger/middle": 0.30775371193885803,
      "test_mae_per_finger/pinky": 0.33731287717819214,
      "test_mae_per_finger/ring": 0.30770373344421387,
      "test_mae_per_finger/thumb": 0.2395099401473999,
      "test_mae_per_pd/distal": 0.19056399166584015,
      "test_mae_per_pd/mid": 0.4308355152606964,
      "test_mae_per_pd/proximal": 0.26911017298698425,
      "test_mae_wrist/deviation": 0.2060348391532898,
      "test_mae_wrist/flexion": 0.31355801224708557
    }
  },
  "production_ready": false,
  "protocol": {
    "adaptations": {
      "declared_devices": [
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "declared_num_workers": 12,
      "declared_per_rank_batch": 600,
      "effective_devices": [
        0
      ],
      "effective_global_batch": 600,
      "effective_num_workers": 0,
      "effective_per_rank_batch": 600,
      "world_size": 1
    },
    "effective": {
      "batch_size": 600,
      "clip_value": 0,
      "global_batch_size": 600,
      "log_every_n_steps": 50,
      "max_epochs": 100,
      "num_workers": 0,
      "precision": "16-mixed",
      "rank": 0,
      "seed": 42,
      "train_shuffle": true,
      "world_size": 1
    },
    "prediction_agreement": "exact_typed_prediction_replay_v1",
    "recipe": {
      "augmentation": "original_resolved_rotation",
      "batch_size": 600,
      "early_stopping_patience": 50,
      "effective_loss_weights": {
        "landmark/fingertip": 0.02,
        "mae": 1.0
      },
      "eta_min": 1e-05,
      "eval_stride": 7790,
      "gradient_clip": 0,
      "jitter": false,
      "layout": {
        "channel_indices": [
          10,
          12,
          0,
          1,
          2,
          4,
          5,
          6
        ],
        "channel_interpolate": false,
        "declared": "emg2pose_interpolate16",
        "effective": "target_hand",
        "input_channels": 8,
        "reason": "official model Conv1d input=8; explicit data-layout correction, network unchanged"
      },
      "lr": 0.001,
      "mask_policy": "official_literal_loader",
      "max_epochs": 100,
      "monitor": "val_loss",
      "monitor_mode": "min",
      "normalization": {
        "asset": "upstream://assets/per_dataset_norm_stats.json",
        "asset_sha256": "b787a1ef98d48f434cb808e0e61e49c1ee0c09fae5b7ca9fc6a05ba4b14ee122",
        "dataset_name": "egoemg",
        "epsilon": 1e-06,
        "estimated_from_project_train_rows": false,
        "field": "filtered",
        "format": "classic_fixed_normalization_v1",
        "key": "egoemg__filtered",
        "limitation": "fixed public pooled asset; its sampling is not proven to match the independent retained training rows",
        "mask_policy": "official_literal_loader",
        "mean": 0.000832,
        "order": "filtered float32 T,C; raw zero-fill if16; pooled scalar; original resolved transforms; C,T",
        "source_revision": "f7cbd048b7092139852270491d3859cd472e0311",
        "std": 8.6551
      },
      "num_workers": 12,
      "precision": "16-mixed",
      "raw_loss_weights": {
        "fingertip_distance": 0.01,
        "landmark/fingertip": 0.01,
        "mae": 1
      },
      "reader_field_selection": {
        "declared_explicit_fields": null,
        "declared_modalities": [
          "emg",
          "joint_angles",
          "labels"
        ],
        "effective_filtered_fields": [
          "emg_left_filtered",
          "emg_right_filtered"
        ],
        "reason": "pinned emg modality omits filtered fields; explicit target-hand field selection enables the original ordinary path"
      },
      "seed": 42,
      "train_stride": 780,
      "upstream_experiment": "emg2pose/regression_vemg2pose_egoemg",
      "val_fraction": 0.12,
      "variant": "vemg2pose",
      "weight_decay": 0.01,
      "window_length": 7790
    },
    "selected_epoch_indexing": "zero-based original checkpoint epoch",
    "test": {
      "batch_size": 600,
      "bitwise_trajectory_guaranteed": false,
      "drop_last": false,
      "final_batch_size": 106,
      "loader_seed": 45,
      "num_workers": 0,
      "precision": "16-mixed",
      "prediction_agreement": "exact_typed_prediction_replay_v1",
      "role": "test",
      "samples": 4306,
      "scientific_eligible": false,
      "scope": "post_termination_original_prediction_pass",
      "seed": 42,
      "shuffle": false,
      "variant": "vemg2pose",
      "world_size": 1
    },
    "test_traversals": 3,
    "validation_prediction_passes": 2
  },
  "runtime": {
    "allow_tf32_cudnn": true,
    "allow_tf32_matmul": true,
    "capability": [
      8,
      0
    ],
    "cuda_build": "11.8",
    "cudnn": 8700,
    "cudnn_benchmark": false,
    "cudnn_deterministic": false,
    "deterministic": false,
    "device_name": "NVIDIA A800-SXM4-80GB",
    "driver_version": "595.91.07",
    "identity_sha256": "a8e8a0fbdfbf36eda3109d05c92c9a704a4627e120b026cc6f430c497838dd8b",
    "matmul_precision": "high",
    "precision_plugin": "MixedPrecision",
    "python": "3.11.12",
    "runtime_profile": "canonical_backend_nondeterministic_v1",
    "scaler_enabled": true,
    "torch_build": "2.2.2+cu118",
    "total_memory": 85094825984,
    "versions": {
      "hydra-core": "1.3.2",
      "joblib": "1.5.2",
      "lightning-utilities": "0.11.9",
      "numpy": "1.26.4",
      "omegaconf": "2.3.1",
      "pandas": "2.2.3",
      "pytorch-lightning": "2.5.0.post0",
      "scipy": "1.13.1",
      "torch": "2.2.2+cu118",
      "torchmetrics": "1.4.3",
      "torchvision": "0.17.2+cu118"
    }
  },
  "scientific_eligible": true,
  "scope": "one_original_classic_experiment_on_released_legacy_data",
  "selection": {
    "best_epoch": 11,
    "best_val_loss": 1.1392862796783447,
    "completed_epochs": 62,
    "last_checkpoint_sha256": "bb5bf74b78eabecb76f42149c21a1c1b2811c2d0b268455c1a7ffee1cd720866",
    "selected_event_sha256": "a9993ad4e92cdd0fff4d62cc1f1188b1eef854c6505227e01b11af6ffba0d3cb",
    "selected_model_sha256": "bd7ac145247a0bef84e1dac7415b778982d8afc3fc1ff40a9e6aa02c2226199c",
    "selected_state_sha256": "c68ccc66ffe0db613d8e1aca1866d29ddcb01c12169057f9a8c410ebce6d3bfd",
    "stop_reason": "early_stopping"
  },
  "sha256": "598726fa45bc46a4c2bfbab34bc65a1fd77ae9d86284c51525ee1213a84efa69",
  "sources_sha256": {
    "configs/neuropose.json": "a1a22af156d3fa1fe60bde71cce0ed63cacbe49049d622e49e714bda6e9800cb",
    "configs/vemg2pose.json": "8d12a4ce68fc6f9f4e3a97777af08fd48fc2b48d5d16a54ecf56cc44dfd05c69",
    "egoemg_repro/__init__.py": "6103cfcd4a90339b804308dcb82a86448749c728fd093ebfab546ff9bfdb3e96",
    "egoemg_repro/classic_emg.py": "cc5a9b838f57fbd6536e61f9aac1cc6328f4b7e5e2da486f7d1abdba4785c1a5",
    "egoemg_repro/classic_emg_artifacts.py": "f68b5441b5dd3f79ee96cdb897dd7b62991357eb95e9bb9502b66a1ff7ecdc0b",
    "egoemg_repro/classic_emg_cohort.py": "72267d660f55e9f5fbbd2e1e17edf07627ef01d2ba09e92fca1086c50a16dbd3",
    "egoemg_repro/classic_emg_cuda_check.py": "cfbccea940016fd1981aeebdfa7144399ba804ced7c66adec81cd7a32e8e8c54",
    "egoemg_repro/classic_emg_data.py": "5d1783045b094e2ac404ea8a3605122e93ef40c5995d4d18a964f7b0bff62e06",
    "egoemg_repro/classic_emg_evaluate.py": "6db6f44afdaa74053407f906523d12b61a6c0ff62f1acb609fd08c938e0759fc",
    "egoemg_repro/classic_emg_formal.py": "9f2ecd61c0731571137a85bd61f75a624fa910c8a0fec3eb454ed4f3ef40ab31",
    "egoemg_repro/classic_emg_report.py": "bdddaf770519151c24cd0f8fdbd9f40e4453e04fa1a28e15088ad825af325868",
    "egoemg_repro/classic_emg_train.py": "52b1ce655f3c4e107e3d809e403cd507fbcfd655f7ac9ea7bc0b9a592b18b0d8",
    "egoemg_repro/data.py": "de7a716c6616277f6189c94abb09803f950dd27bc1f9f7193b1a000758a88f0e",
    "egoemg_repro/metrics.py": "12d6eb170997ae78ddbf5602d1252d05a778f1357f17b28aa3b96d06a618798e",
    "egoemg_repro/partitions.py": "ec816483e5d567cec45df77bd6823e97c201d3fd20f45b0d8004827565fdf678",
    "tools/classic_capacity.py": "b890de97c1ebafcb5ddb88dde91785a40a439dce904e919e433f82f8389f7d87"
  },
  "statistics": {
    "alignment": {
      "input_length": 7790,
      "left_context": 1790,
      "original_center": 3895,
      "output_center": 2105,
      "output_length": 6000,
      "right_context": 0
    },
    "center": {
      "by_component": {
        "fingers": {
          "absolute_error_sum_rad": 24048.045626046216,
          "mae_deg": 16.489367160249973,
          "valid_angle_count": 83560
        },
        "wrist": {
          "absolute_error_sum_rad": 2177.5445680349276,
          "mae_deg": 14.931081073484837,
          "valid_angle_count": 8356
        }
      },
      "by_finger": {
        "index": {
          "absolute_error_sum_rad": 4260.146690815687,
          "mae_deg": 14.60557835629266,
          "valid_angle_count": 16712
        },
        "middle": {
          "absolute_error_sum_rad": 5098.514037136498,
          "mae_deg": 17.479854961472473,
          "valid_angle_count": 16712
        },
        "pinky": {
          "absolute_error_sum_rad": 5592.845100564315,
          "mae_deg": 19.174630189848955,
          "valid_angle_count": 16712
        },
        "ring": {
          "absolute_error_sum_rad": 5114.134302299373,
          "mae_deg": 17.533407813836522,
          "valid_angle_count": 16712
        },
        "thumb": {
          "absolute_error_sum_rad": 3982.405495230343,
          "mae_deg": 13.653364479799254,
          "valid_angle_count": 16712
        }
      },
      "by_hand": {
        "left": {
          "absolute_error_sum_rad": 13059.880350180745,
          "mae_deg": 16.227360014989426,
          "valid_angle_count": 46112
        },
        "right": {
          "absolute_error_sum_rad": 13165.709843900397,
          "mae_deg": 16.468858797252086,
          "valid_angle_count": 45804
        }
      },
      "by_hand_split": {
        "left": {
          "both": {
            "absolute_error_sum_rad": 1024.1918871811067,
            "mae_deg": 17.09844771183461,
            "valid_angle_count": 3432
          },
          "gesture": {
            "absolute_error_sum_rad": 6933.808766417056,
            "mae_deg": 16.094554296974113,
            "valid_angle_count": 24684
          },
          "user": {
            "absolute_error_sum_rad": 5101.879696582583,
            "mae_deg": 16.243397099225774,
            "valid_angle_count": 17996
          }
        },
        "right": {
          "both": {
            "absolute_error_sum_rad": 1032.3046980091804,
            "mae_deg": 17.124117651260285,
            "valid_angle_count": 3454
          },
          "gesture": {
            "absolute_error_sum_rad": 6876.8558488817,
            "mae_deg": 16.134922868991662,
            "valid_angle_count": 24420
          },
          "user": {
            "absolute_error_sum_rad": 5256.549297009517,
            "mae_deg": 16.79743946018434,
            "valid_angle_count": 17930
          }
        }
      },
      "by_joint": [
        {
          "absolute_error_sum_rad": 892.961698160716,
          "index": 0,
          "mae_deg": 12.245796211451399,
          "name": "THUMB_CMC_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 1109.268311646967,
          "index": 1,
          "mae_deg": 15.21215715772468,
          "name": "THUMB_CMC_AA",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 1013.8071578601375,
          "index": 2,
          "mae_deg": 13.903032883087382,
          "name": "THUMB_MCP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 966.3683275625226,
          "index": 3,
          "mae_deg": 13.252471666933559,
          "name": "THUMB_IP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 583.4837490808277,
          "index": 4,
          "mae_deg": 8.001712837913296,
          "name": "INDEX_MCP_AA",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 1264.5824469873041,
          "index": 5,
          "mae_deg": 17.342086419027932,
          "name": "INDEX_MCP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 1739.9610999874421,
          "index": 6,
          "mae_deg": 23.86127992968422,
          "name": "INDEX_PIP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 672.1193947601132,
          "index": 7,
          "mae_deg": 9.217234238545185,
          "name": "INDEX_DIP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 442.9235593219346,
          "index": 8,
          "mae_deg": 6.0741145514741985,
          "name": "MIDDLE_MCP_AA",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 1684.7557048290182,
          "index": 9,
          "mae_deg": 23.104210482836535,
          "name": "MIDDLE_MCP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 2058.431935987086,
          "index": 10,
          "mae_deg": 28.228688929392835,
          "name": "MIDDLE_PIP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 912.402836998459,
          "index": 11,
          "mae_deg": 12.512405882186332,
          "name": "MIDDLE_DIP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 419.3608610421559,
          "index": 12,
          "mae_deg": 5.750983109307735,
          "name": "RING_MCP_AA",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 1717.1846949798346,
          "index": 13,
          "mae_deg": 23.54893146644426,
          "name": "RING_MCP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 2298.3997971010394,
          "index": 14,
          "mae_deg": 31.519532792631477,
          "name": "RING_PIP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 679.1889491763432,
          "index": 15,
          "mae_deg": 9.314183886962622,
          "name": "RING_DIP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 905.2976150082541,
          "index": 16,
          "mae_deg": 12.4149670998641,
          "name": "PINKY_MCP_AA",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 2123.306905280173,
          "index": 17,
          "mae_deg": 29.118363878300137,
          "name": "PINKY_MCP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 1845.293158594286,
          "index": 18,
          "mae_deg": 25.305770692153533,
          "name": "PINKY_PIP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 718.9474216816016,
          "index": 19,
          "mae_deg": 9.859419089078044,
          "name": "PINKY_DIP_FE",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 1307.3294899895918,
          "index": 20,
          "mae_deg": 17.928305938102923,
          "name": "wrist_pitch",
          "valid_angle_count": 4178
        },
        {
          "absolute_error_sum_rad": 870.2150780453358,
          "index": 21,
          "mae_deg": 11.93385620886675,
          "name": "wrist_yaw",
          "valid_angle_count": 4178
        }
      ],
      "by_pd_group": {
        "distal": {
          "absolute_error_sum_rad": 3949.0269301790395,
          "mae_deg": 10.831142952741148,
          "valid_angle_count": 20890
        },
        "mid": {
          "absolute_error_sum_rad": 8955.893149529991,
          "mae_deg": 24.56366104538989,
          "valid_angle_count": 20890
        },
        "proximal": {
          "absolute_error_sum_rad": 11143.125546337185,
          "mae_deg": 15.281332321434427,
          "valid_angle_count": 41780
        }
      },
      "by_split": {
        "both": {
          "absolute_error_sum_rad": 2056.496585190287,
          "mae_deg": 17.11132368784046,
          "valid_angle_count": 6886
        },
        "gesture": {
          "absolute_error_sum_rad": 13810.664615298756,
          "mae_deg": 16.114630065316177,
          "valid_angle_count": 49104
        },
        "user": {
          "absolute_error_sum_rad": 10358.4289935921,
          "mae_deg": 16.519909361431058,
          "valid_angle_count": 35926
        }
      },
      "by_user": {
        "participant_001": {
          "absolute_error_sum_rad": 380.0113515453413,
          "mae_deg": 17.672927443677185,
          "valid_angle_count": 1232
        },
        "participant_002": {
          "absolute_error_sum_rad": 521.439534440171,
          "mae_deg": 17.410422257976883,
          "valid_angle_count": 1716
        },
        "participant_003": {
          "absolute_error_sum_rad": 390.3659948614659,
          "mae_deg": 15.40380438773249,
          "valid_angle_count": 1452
        },
        "participant_004": {
          "absolute_error_sum_rad": 9.492615175782703,
          "mae_deg": 12.36106332077693,
          "valid_angle_count": 44
        },
        "participant_005": {
          "absolute_error_sum_rad": 351.94835234823404,
          "mae_deg": 14.549174023186888,
          "valid_angle_count": 1386
        },
        "participant_006": {
          "absolute_error_sum_rad": 427.4660963607748,
          "mae_deg": 18.250374967514873,
          "valid_angle_count": 1342
        },
        "participant_007": {
          "absolute_error_sum_rad": 451.37261909488006,
          "mae_deg": 15.885593404117161,
          "valid_angle_count": 1628
        },
        "participant_008": {
          "absolute_error_sum_rad": 412.69141167297494,
          "mae_deg": 14.92769957711961,
          "valid_angle_count": 1584
        },
        "participant_009": {
          "absolute_error_sum_rad": 516.4250648411689,
          "mae_deg": 16.011351001173622,
          "valid_angle_count": 1848
        },
        "participant_010": {
          "absolute_error_sum_rad": 557.9947902224376,
          "mae_deg": 17.096655866328025,
          "valid_angle_count": 1870
        },
        "participant_011": {
          "absolute_error_sum_rad": 361.7480129223404,
          "mae_deg": 15.195479756373953,
          "valid_angle_count": 1364
        },
        "participant_012": {
          "absolute_error_sum_rad": 555.3815265865414,
          "mae_deg": 16.436475977759063,
          "valid_angle_count": 1936
        },
        "participant_013": {
          "absolute_error_sum_rad": 380.5789202400556,
          "mae_deg": 17.089001490126307,
          "valid_angle_count": 1276
        },
        "participant_014": {
          "absolute_error_sum_rad": 3402.3793522249325,
          "mae_deg": 16.348706573712143,
          "valid_angle_count": 11924
        },
        "participant_015": {
          "absolute_error_sum_rad": 314.55156374780927,
          "mae_deg": 17.066739623096343,
          "valid_angle_count": 1056
        },
        "participant_016": {
          "absolute_error_sum_rad": 495.8263528038224,
          "mae_deg": 16.34566017665338,
          "valid_angle_count": 1738
        },
        "participant_017": {
          "absolute_error_sum_rad": 539.8873540747445,
          "mae_deg": 17.575719773277257,
          "valid_angle_count": 1760
        },
        "participant_018": {
          "absolute_error_sum_rad": 46.179550404660404,
          "mae_deg": 15.033484874992581,
          "valid_angle_count": 176
        },
        "participant_019": {
          "absolute_error_sum_rad": 194.48159543797374,
          "mae_deg": 16.883294866010033,
          "valid_angle_count": 660
        },
        "participant_020": {
          "absolute_error_sum_rad": 561.0621122140437,
          "mae_deg": 14.91024632400747,
          "valid_angle_count": 2156
        },
        "participant_021": {
          "absolute_error_sum_rad": 470.97294376202626,
          "mae_deg": 15.142964052983377,
          "valid_angle_count": 1782
        },
        "participant_022": {
          "absolute_error_sum_rad": 561.7582797351351,
          "mae_deg": 14.928747001555067,
          "valid_angle_count": 2156
        },
        "participant_023": {
          "absolute_error_sum_rad": 222.38995589292608,
          "mae_deg": 19.3060695132806,
          "valid_angle_count": 660
        },
        "participant_024": {
          "absolute_error_sum_rad": 459.7413186203339,
          "mae_deg": 14.253916247145227,
          "valid_angle_count": 1848
        },
        "participant_025": {
          "absolute_error_sum_rad": 436.8978782061604,
          "mae_deg": 15.80328566883438,
          "valid_angle_count": 1584
        },
        "participant_026": {
          "absolute_error_sum_rad": 2695.619413733304,
          "mae_deg": 17.909046333539845,
          "valid_angle_count": 8624
        },
        "participant_027": {
          "absolute_error_sum_rad": 251.54582924599526,
          "mae_deg": 15.597959274798702,
          "valid_angle_count": 924
        },
        "participant_028": {
          "absolute_error_sum_rad": 609.6542636496597,
          "mae_deg": 17.64172538852599,
          "valid_angle_count": 1980
        },
        "participant_029": {
          "absolute_error_sum_rad": 594.5714677666547,
          "mae_deg": 17.59629944314462,
          "valid_angle_count": 1936
        },
        "participant_030": {
          "absolute_error_sum_rad": 284.2623636071803,
          "mae_deg": 14.806394281004218,
          "valid_angle_count": 1100
        },
        "participant_031": {
          "absolute_error_sum_rad": 277.2654211926274,
          "mae_deg": 14.441944035686076,
          "valid_angle_count": 1100
        },
        "participant_032": {
          "absolute_error_sum_rad": 1474.9921860270315,
          "mae_deg": 15.243655677150812,
          "valid_angle_count": 5544
        },
        "participant_033": {
          "absolute_error_sum_rad": 115.02746706455946,
          "mae_deg": 14.265342837401162,
          "valid_angle_count": 462
        },
        "participant_034": {
          "absolute_error_sum_rad": 177.85396813647822,
          "mae_deg": 15.439820824052159,
          "valid_angle_count": 660
        },
        "participant_035": {
          "absolute_error_sum_rad": 277.84022510268187,
          "mae_deg": 16.445322600566552,
          "valid_angle_count": 968
        },
        "participant_036": {
          "absolute_error_sum_rad": 258.90617864526575,
          "mae_deg": 14.658331350033446,
          "valid_angle_count": 1012
        },
        "participant_037": {
          "absolute_error_sum_rad": 298.09585794829763,
          "mae_deg": 17.644250568976002,
          "valid_angle_count": 968
        },
        "participant_038": {
          "absolute_error_sum_rad": 391.1855531136098,
          "mae_deg": 16.432024340103034,
          "valid_angle_count": 1364
        },
        "participant_039": {
          "absolute_error_sum_rad": 2659.209136533347,
          "mae_deg": 16.257091374945443,
          "valid_angle_count": 9372
        },
        "participant_040": {
          "absolute_error_sum_rad": 2423.5956863436077,
          "mae_deg": 16.437240065530393,
          "valid_angle_count": 8448
        },
        "participant_041": {
          "absolute_error_sum_rad": 412.9206285341061,
          "mae_deg": 18.54122985023002,
          "valid_angle_count": 1276
        }
      },
      "invalid_angle_count": 2816,
      "mode": "center",
      "overall": {
        "absolute_error_sum_rad": 26225.59019408114,
        "mae_deg": 16.347704788725867,
        "valid_angle_count": 91916
      },
      "user_summary": {
        "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
        "mean_mae_deg": 16.127964302807303,
        "std_mae_deg": 1.3997933799703715,
        "user_count": 41
      }
    },
    "samples": 4306,
    "trajectory": {
      "by_component": {
        "fingers": {
          "absolute_error_sum_rad": 145243912.98491505,
          "mae_deg": 16.608186238107365,
          "valid_angle_count": 501069960
        },
        "wrist": {
          "absolute_error_sum_rad": 13030869.391052688,
          "mae_deg": 14.90039074570601,
          "valid_angle_count": 50106996
        }
      },
      "by_finger": {
        "index": {
          "absolute_error_sum_rad": 25778098.717537567,
          "mae_deg": 14.738223983598045,
          "valid_angle_count": 100213992
        },
        "middle": {
          "absolute_error_sum_rad": 30818190.12307617,
          "mae_deg": 17.619817263481767,
          "valid_angle_count": 100213992
        },
        "pinky": {
          "absolute_error_sum_rad": 33803316.99257037,
          "mae_deg": 19.326516772399817,
          "valid_angle_count": 100213992
        },
        "ring": {
          "absolute_error_sum_rad": 30829247.069420107,
          "mae_deg": 17.626138899284978,
          "valid_angle_count": 100213992
        },
        "thumb": {
          "absolute_error_sum_rad": 24015060.082310434,
          "mae_deg": 13.73023427177198,
          "valid_angle_count": 100213992
        }
      },
      "by_hand": {
        "left": {
          "absolute_error_sum_rad": 78863408.52210082,
          "mae_deg": 16.369551253674416,
          "valid_angle_count": 276033252
        },
        "right": {
          "absolute_error_sum_rad": 79411373.85386659,
          "mae_deg": 16.536582524025665,
          "valid_angle_count": 275143704
        }
      },
      "by_hand_split": {
        "left": {
          "both": {
            "absolute_error_sum_rad": 6150237.703048595,
            "mae_deg": 17.201696168095502,
            "valid_angle_count": 20485344
          },
          "gesture": {
            "absolute_error_sum_rad": 41600634.437083706,
            "mae_deg": 16.136818779019123,
            "valid_angle_count": 147708220
          },
          "user": {
            "absolute_error_sum_rad": 31112536.381968655,
            "mae_deg": 16.53025020467444,
            "valid_angle_count": 107839688
          }
        },
        "right": {
          "both": {
            "absolute_error_sum_rad": 6240059.87087308,
            "mae_deg": 17.235376286822348,
            "valid_angle_count": 20743910
          },
          "gesture": {
            "absolute_error_sum_rad": 41306239.349009894,
            "mae_deg": 16.134248491672178,
            "valid_angle_count": 146686298
          },
          "user": {
            "absolute_error_sum_rad": 31865074.63398336,
            "mae_deg": 16.949912111260623,
            "valid_angle_count": 107713496
          }
        }
      },
      "by_joint": [
        {
          "absolute_error_sum_rad": 5407347.320820659,
          "index": 0,
          "mae_deg": 12.366264377309585,
          "name": "THUMB_CMC_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 6684130.9794702735,
          "index": 1,
          "mae_deg": 15.286188572800961,
          "name": "THUMB_CMC_AA",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 6112314.653193116,
          "index": 2,
          "mae_deg": 13.978480477414173,
          "name": "THUMB_MCP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 5811267.1288263835,
          "index": 3,
          "mae_deg": 13.290003659563206,
          "name": "THUMB_IP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 3484436.016387713,
          "index": 4,
          "mae_deg": 7.968686756731269,
          "name": "INDEX_MCP_AA",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 7701025.766405224,
          "index": 5,
          "mae_deg": 17.61176320913429,
          "name": "INDEX_MCP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 10511571.959827391,
          "index": 6,
          "mae_deg": 24.03930618176228,
          "name": "INDEX_PIP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 4081064.9749172414,
          "index": 7,
          "mae_deg": 9.33313978676435,
          "name": "INDEX_DIP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 2663575.2979959454,
          "index": 8,
          "mae_deg": 6.091429747273943,
          "name": "MIDDLE_MCP_AA",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 10226803.967605341,
          "index": 9,
          "mae_deg": 23.388059633486353,
          "name": "MIDDLE_MCP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 12433076.681029566,
          "index": 10,
          "mae_deg": 28.433667034659802,
          "name": "MIDDLE_PIP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 5494734.176445321,
          "index": 11,
          "mae_deg": 12.566112638506969,
          "name": "MIDDLE_DIP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 2524724.03922946,
          "index": 12,
          "mae_deg": 5.773885622002554,
          "name": "RING_MCP_AA",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 10408906.035324106,
          "index": 13,
          "mae_deg": 23.8045156477679,
          "name": "RING_MCP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 13767826.91067817,
          "index": 14,
          "mae_deg": 31.48615714454316,
          "name": "RING_PIP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 4127790.084188371,
          "index": 15,
          "mae_deg": 9.439997182826303,
          "name": "RING_DIP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 5429890.40820575,
          "index": 16,
          "mae_deg": 12.417819005104887,
          "name": "PINKY_MCP_AA",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 12852747.356008327,
          "index": 17,
          "mae_deg": 29.39342756237891,
          "name": "PINKY_MCP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 11180802.494870974,
          "index": 18,
          "mae_deg": 25.56979446644329,
          "name": "PINKY_PIP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 4339876.733485321,
          "index": 19,
          "mae_deg": 9.92502605567218,
          "name": "PINKY_DIP_FE",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 7863246.620423971,
          "index": 20,
          "mae_deg": 17.982752133885718,
          "name": "wrist_pitch",
          "valid_angle_count": 25053498
        },
        {
          "absolute_error_sum_rad": 5167622.770628706,
          "index": 21,
          "mae_deg": 11.81802935752628,
          "name": "wrist_yaw",
          "valid_angle_count": 25053498
        }
      ],
      "by_pd_group": {
        "distal": {
          "absolute_error_sum_rad": 23854733.09786264,
          "mae_deg": 10.910855864666603,
          "valid_angle_count": 125267490
        },
        "mid": {
          "absolute_error_sum_rad": 54005592.69959922,
          "mae_deg": 24.701481060964547,
          "valid_angle_count": 125267490
        },
        "proximal": {
          "absolute_error_sum_rad": 67383587.1874528,
          "mae_deg": 15.410204013399063,
          "valid_angle_count": 250534980
        }
      },
      "by_split": {
        "both": {
          "absolute_error_sum_rad": 12390297.57392168,
          "mae_deg": 17.218641838557044,
          "valid_angle_count": 41229254
        },
        "gesture": {
          "absolute_error_sum_rad": 82906873.78609367,
          "mae_deg": 16.135538096422593,
          "valid_angle_count": 294394518
        },
        "user": {
          "absolute_error_sum_rad": 62977611.01595198,
          "mae_deg": 16.73995831595162,
          "valid_angle_count": 215553184
        }
      },
      "by_user": {
        "participant_001": {
          "absolute_error_sum_rad": 2188473.8736101636,
          "mae_deg": 16.96297572138922,
          "valid_angle_count": 7392000
        },
        "participant_002": {
          "absolute_error_sum_rad": 2945950.480108052,
          "mae_deg": 16.393796538920935,
          "valid_angle_count": 10296000
        },
        "participant_003": {
          "absolute_error_sum_rad": 2326327.6991032693,
          "mae_deg": 15.337711135472643,
          "valid_angle_count": 8690264
        },
        "participant_004": {
          "absolute_error_sum_rad": 40908.95705656055,
          "mae_deg": 8.878449180389572,
          "valid_angle_count": 264000
        },
        "participant_005": {
          "absolute_error_sum_rad": 2061416.4855807645,
          "mae_deg": 14.321002653269474,
          "valid_angle_count": 8247360
        },
        "participant_006": {
          "absolute_error_sum_rad": 2604486.6078576837,
          "mae_deg": 18.57838113839539,
          "valid_angle_count": 8032244
        },
        "participant_007": {
          "absolute_error_sum_rad": 2660501.030978415,
          "mae_deg": 15.552704710767381,
          "valid_angle_count": 9801220
        },
        "participant_008": {
          "absolute_error_sum_rad": 2471403.6710138423,
          "mae_deg": 14.899095099140505,
          "valid_angle_count": 9504000
        },
        "participant_009": {
          "absolute_error_sum_rad": 3136286.1140912413,
          "mae_deg": 16.232433228991173,
          "valid_angle_count": 11070180
        },
        "participant_010": {
          "absolute_error_sum_rad": 3276694.3663231404,
          "mae_deg": 16.931147480991058,
          "valid_angle_count": 11088484
        },
        "participant_011": {
          "absolute_error_sum_rad": 2211641.701727747,
          "mae_deg": 15.781793784379278,
          "valid_angle_count": 8029362
        },
        "participant_012": {
          "absolute_error_sum_rad": 3204510.601726925,
          "mae_deg": 15.809683126310174,
          "valid_angle_count": 11613448
        },
        "participant_013": {
          "absolute_error_sum_rad": 2295454.1257931213,
          "mae_deg": 17.360245814876322,
          "valid_angle_count": 7575920
        },
        "participant_014": {
          "absolute_error_sum_rad": 20628793.83980374,
          "mae_deg": 16.56812301344589,
          "valid_angle_count": 71338366
        },
        "participant_015": {
          "absolute_error_sum_rad": 1981001.4760936068,
          "mae_deg": 17.916600178016793,
          "valid_angle_count": 6335076
        },
        "participant_016": {
          "absolute_error_sum_rad": 3110184.4537089188,
          "mae_deg": 16.87975418750638,
          "valid_angle_count": 10557052
        },
        "participant_017": {
          "absolute_error_sum_rad": 3232180.064152728,
          "mae_deg": 17.536957983169977,
          "valid_angle_count": 10560000
        },
        "participant_018": {
          "absolute_error_sum_rad": 304886.33062132454,
          "mae_deg": 16.542329522568306,
          "valid_angle_count": 1056000
        },
        "participant_019": {
          "absolute_error_sum_rad": 1185157.0225318605,
          "mae_deg": 16.847986501504465,
          "valid_angle_count": 4030422
        },
        "participant_020": {
          "absolute_error_sum_rad": 3479627.3988383487,
          "mae_deg": 15.411871075411424,
          "valid_angle_count": 12936000
        },
        "participant_021": {
          "absolute_error_sum_rad": 2706616.7693478684,
          "mae_deg": 14.382840877600543,
          "valid_angle_count": 10782134
        },
        "participant_022": {
          "absolute_error_sum_rad": 3435141.536487905,
          "mae_deg": 15.214835503311829,
          "valid_angle_count": 12936000
        },
        "participant_023": {
          "absolute_error_sum_rad": 1295100.1030907363,
          "mae_deg": 19.029691466426613,
          "valid_angle_count": 3899368
        },
        "participant_024": {
          "absolute_error_sum_rad": 2831257.3038621093,
          "mae_deg": 14.630149190736592,
          "valid_angle_count": 11088000
        },
        "participant_025": {
          "absolute_error_sum_rad": 2621852.578739224,
          "mae_deg": 15.806090831991675,
          "valid_angle_count": 9504000
        },
        "participant_026": {
          "absolute_error_sum_rad": 16093061.658694299,
          "mae_deg": 17.833643791486672,
          "valid_angle_count": 51703652
        },
        "participant_027": {
          "absolute_error_sum_rad": 1515711.051380544,
          "mae_deg": 15.664474423780995,
          "valid_angle_count": 5544000
        },
        "participant_028": {
          "absolute_error_sum_rad": 3726970.3096092464,
          "mae_deg": 17.974719622152826,
          "valid_angle_count": 11880000
        },
        "participant_029": {
          "absolute_error_sum_rad": 3518230.21676052,
          "mae_deg": 17.219200823715287,
          "valid_angle_count": 11706684
        },
        "participant_030": {
          "absolute_error_sum_rad": 1713707.798430023,
          "mae_deg": 14.877003661923686,
          "valid_angle_count": 6600000
        },
        "participant_031": {
          "absolute_error_sum_rad": 1662760.7735751627,
          "mae_deg": 14.434723434206775,
          "valid_angle_count": 6600000
        },
        "participant_032": {
          "absolute_error_sum_rad": 9042266.77100231,
          "mae_deg": 15.574907503902692,
          "valid_angle_count": 33264000
        },
        "participant_033": {
          "absolute_error_sum_rad": 694965.1118379122,
          "mae_deg": 14.840519361256883,
          "valid_angle_count": 2683098
        },
        "participant_034": {
          "absolute_error_sum_rad": 1069444.076849619,
          "mae_deg": 15.473391926451423,
          "valid_angle_count": 3960000
        },
        "participant_035": {
          "absolute_error_sum_rad": 1651524.5423737175,
          "mae_deg": 16.292249662584144,
          "valid_angle_count": 5808000
        },
        "participant_036": {
          "absolute_error_sum_rad": 1605086.0207350682,
          "mae_deg": 15.14569412772845,
          "valid_angle_count": 6072000
        },
        "participant_037": {
          "absolute_error_sum_rad": 1833846.9103801546,
          "mae_deg": 18.21949609742628,
          "valid_angle_count": 5766992
        },
        "participant_038": {
          "absolute_error_sum_rad": 2339354.3970242226,
          "mae_deg": 16.24871938311535,
          "valid_angle_count": 8248966
        },
        "participant_039": {
          "absolute_error_sum_rad": 16170397.91403297,
          "mae_deg": 16.43059964299832,
          "valid_angle_count": 56388420
        },
        "participant_040": {
          "absolute_error_sum_rad": 14973876.435063755,
          "mae_deg": 16.92589809579293,
          "valid_angle_count": 50688000
        },
        "participant_041": {
          "absolute_error_sum_rad": 2427723.795968477,
          "mae_deg": 18.215542527513936,
          "valid_angle_count": 7636244
        }
      },
      "invalid_angle_count": 17215044,
      "mode": "trajectory",
      "overall": {
        "absolute_error_sum_rad": 158274782.37596723,
        "mae_deg": 16.452932102434463,
        "valid_angle_count": 551176956
      },
      "user_summary": {
        "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
        "mean_mae_deg": 16.12627887880537,
        "std_mae_deg": 1.6694038741311266,
        "user_count": 41
      }
    }
  },
  "training": {
    "attempts": 10540,
    "bitwise_trajectory_guaranteed": false,
    "completed_epochs": 62,
    "interrupted": false,
    "natural_overflows": 5,
    "optimizer_executions": 10535,
    "runtime_profile": "canonical_backend_nondeterministic_v1",
    "selection_scope": "terminated-full-validation-loss",
    "stop_reason": "early_stopping",
    "trajectory_agreement_qualified": false
  },
  "variant": "vemg2pose"
}
```

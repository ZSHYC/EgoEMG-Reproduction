# ResNet-18 + EMGFormer-S 融合中心评价

完整训练实验

模型：fusion_rn18_s_center_8ch；训练完成 150 轮，实际更新 46950 次。

训练预算：完整实验150轮、不早停；工程实验显式限于两轮。

验证选择：rn18-s-8ch-centerfusion-epoch=144-val_mae=0.1063.ckpt；不以最后一轮替代验证最优。

测试消费 2792 手样本，唯一 2792，重复消费 0。

| 统计人口 | MAE (°) | 有效角度数 | 弧度绝对误差和 |
| --- | ---: | ---: | ---: |
| 全部请求中心 | 6.189098 | 59576 | 6435.407252 |
| 唯一中心 | 6.189098 | 59576 | 6435.407252 |
| 全部请求中的有图子集 | 6.079778 | 59136 | 6275.047437 |

原生按样本数加权的批量 MAE：0.107943289 rad。

该指标与按有效角度数汇总的 pooled MAE 分开报告。缺图仍保留原生成标签 mask；不增加腕角过滤。

采用可用旧版 EgoEMG 队列、独立 train 留出验证、单 rank，正式实验每批200/10worker；未证明与论文原表或缺失的合并训练来源数值等价。
当前报告只验收所列实验，整篇论文复现仍未完成。

## 完整公开证据

以下与 metrics.json 一致，包含逐手、逐条件、匿名逐用户、逐角、指组、用户交集与来源摘要。

```json
{
  "cohort_sha256": "1736318c9faf5740417ca676cd520d47ada84d16eac31f865e9cdf24a9f2e38c",
  "curve_data": {
    "epochs": [
      {
        "epoch": 0,
        "global_step": 313,
        "model_sha256": "16ca24b1231c5eafa6490af65212cea20c0148968d402bf205288b6d38ea48c7",
        "optimizer_executions": 313,
        "train_batches": 313,
        "train_loss": 0.5119601913894942,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "7268a0e020031aad41aa265728aba96db2596cb69507ab2eb8250d0478ef8178",
          "prediction_sha256": "a7f057474c02880adf498c76d2c595158c61adea38146fb45990c13ff9f97294",
          "samples": 1052,
          "val_mae": 0.16656197607517242
        }
      },
      {
        "epoch": 1,
        "global_step": 626,
        "model_sha256": "1f91b80a0d835b447ac56831e1288f704943f86b8828e6bd5f7b04f189a5b6b5",
        "optimizer_executions": 626,
        "train_batches": 313,
        "train_loss": 0.4395242985370771,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "e75620b8eda45f919f68fcaa05580f6cf225f02fd66b37ffd9e8d78a3bf5fb90",
          "prediction_sha256": "b076957ff5f3023e24522b31f95c3d1ce156e62a87a874bbacd268d153ccf72e",
          "samples": 1052,
          "val_mae": 0.1450224369764328
        }
      },
      {
        "epoch": 2,
        "global_step": 939,
        "model_sha256": "b56b8f0cefd170f6e5cf956634a71767384435e5356ac7c79fa1e74d058eb479",
        "optimizer_executions": 939,
        "train_batches": 313,
        "train_loss": 0.41191877422093087,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "a5c1c6b29b66ebf58bd617d26b034f98fdb50b3989a53b997cf9414516b0b164",
          "prediction_sha256": "59f1eca4457d5ae36a6a6ccde6bf1936fd9eb6334cfd690bf35a9987f90262f1",
          "samples": 1052,
          "val_mae": 0.13948678970336914
        }
      },
      {
        "epoch": 3,
        "global_step": 1252,
        "model_sha256": "08f3bb2ef8fac173705d711d98d160bbff35c36651421013245d42ab945558d3",
        "optimizer_executions": 1252,
        "train_batches": 313,
        "train_loss": 0.3955032290869643,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "68e3a212f25fb2fe0a7181c9c1e2f17a5c827c6b9bf90d675b09eec2f2a31ed5",
          "prediction_sha256": "00ada9f66bb37d80c2b87fc019c51e2cfb3b6a0651bb3c6df5116a8c04038764",
          "samples": 1052,
          "val_mae": 0.13836567103862762
        }
      },
      {
        "epoch": 4,
        "global_step": 1565,
        "model_sha256": "3671287ac35f5506129f1419b950a1925a8b6ce069774c14fe3419d95fc395fd",
        "optimizer_executions": 1565,
        "train_batches": 313,
        "train_loss": 0.3814264010701806,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "f623456d97485758a9fb0a265a8b49e490cef87d651f0a1345a789772f00559a",
          "prediction_sha256": "283dc39b3b34917e231b18f8353ac24cd73863c230a401391c8234ad24971291",
          "samples": 1052,
          "val_mae": 0.1327221393585205
        }
      },
      {
        "epoch": 5,
        "global_step": 1878,
        "model_sha256": "a078305df2921f78485463b5981336d302f4eedff650e721d410f1f3d2acc3b4",
        "optimizer_executions": 1878,
        "train_batches": 313,
        "train_loss": 0.37104083306888247,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "399a8f5fee729b7718d312391e89a24cc6f4caf394c616d7a6dfc1269c83ed4f",
          "prediction_sha256": "0fa0550a5a33e928fbd25388e5b464ef36f2eb1c88901b659752cbc7cbaf45f8",
          "samples": 1052,
          "val_mae": 0.13347415626049042
        }
      },
      {
        "epoch": 6,
        "global_step": 2191,
        "model_sha256": "8d6f59f1d7e02feb07e30811a50a35b460073489ba6cf928940ccb153bac6c94",
        "optimizer_executions": 2191,
        "train_batches": 313,
        "train_loss": 0.3616312031404713,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "5bd91b66b9ccedfb704e3d814a2e80fd908fe2a2b54b68c6e31a7e9f4b3d7012",
          "prediction_sha256": "4ea8227c85617c88a27a5aa912934cd848b9c2770f71e4a576ef363100ebff38",
          "samples": 1052,
          "val_mae": 0.133224755525589
        }
      },
      {
        "epoch": 7,
        "global_step": 2504,
        "model_sha256": "042e64564b6374956a9e4a6e13935def419a3762db9deb3abb6ab150254ef0b7",
        "optimizer_executions": 2504,
        "train_batches": 313,
        "train_loss": 0.354620828730823,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "f4ad5ee3b50ec5024dcb69458814e59fdb1c3a01e330f6aaae06aac204eb440b",
          "prediction_sha256": "a35ee161b19ed0e846da0bdc422e2722cfd2945bd9a4eaae074cea4398fced1e",
          "samples": 1052,
          "val_mae": 0.13324646651744843
        }
      },
      {
        "epoch": 8,
        "global_step": 2817,
        "model_sha256": "339919d53ec47710a4859734577e4d32ebd03d92c429e7d3fa375f56b17e2b61",
        "optimizer_executions": 2817,
        "train_batches": 313,
        "train_loss": 0.34701213084145777,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "755ecf38df3ba65c80bb01e509156ac46090e5ac963f2f6d8415d3720610e8ac",
          "prediction_sha256": "84dfe02aa1e2215308a423ab72ec0bba70df07e900e68605b3036595d6460010",
          "samples": 1052,
          "val_mae": 0.1305212676525116
        }
      },
      {
        "epoch": 9,
        "global_step": 3130,
        "model_sha256": "209929f536b5447b8f022b5009322c9d61d5965139e6a71eeb92cb5f84cb9a90",
        "optimizer_executions": 3130,
        "train_batches": 313,
        "train_loss": 0.3423474871656199,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "5a64e5276f240a4127d1226b11f901a313d47677004de24a23cd1193582b2e03",
          "prediction_sha256": "27d8dd17f873f556e278f2e969fccfda5c20fefc281de3fc4b372e10ec758568",
          "samples": 1052,
          "val_mae": 0.12980227172374725
        }
      },
      {
        "epoch": 10,
        "global_step": 3443,
        "model_sha256": "0f954160d715dccf14ff4587e8993c0ad86303ae590e623b95abfcf218f1760b",
        "optimizer_executions": 3443,
        "train_batches": 313,
        "train_loss": 0.3359939548107497,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "d02dfd62ce7bf7d331d208bf1c670c5e3c0c8e2818aee555ca638d7485ca788e",
          "prediction_sha256": "0074fef6921b52aba99fa8396cdc78ceecfacfb2f2378ef1f06de6a0cca91190",
          "samples": 1052,
          "val_mae": 0.12915126979351044
        }
      },
      {
        "epoch": 11,
        "global_step": 3756,
        "model_sha256": "72a58fe0431be735b8e6ccb74a46be445690bdbef2bcac90a35e98b1d9838995",
        "optimizer_executions": 3756,
        "train_batches": 313,
        "train_loss": 0.33265260756445136,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "31370ccd1fc70afeac10034249e077e80475744ef36b2809471e52b6809bba4e",
          "prediction_sha256": "58324d3592c24deafe66cb80804d990f062552c32d5ce3d764a7f930f2ee3b8b",
          "samples": 1052,
          "val_mae": 0.12434889376163483
        }
      },
      {
        "epoch": 12,
        "global_step": 4069,
        "model_sha256": "944252eb0a2f1bcd27116cdd5aec79e19a5cfdfc21e1354602a98acea1928fc2",
        "optimizer_executions": 4069,
        "train_batches": 313,
        "train_loss": 0.33132645322243387,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "03e884f22c64e962293be1639866e9bd6176eddf39d534c59f19333423d4232a",
          "prediction_sha256": "78db0527e365ffc6fbb842aa7dc2bc38b9c7435d8acbc8e7c317a67448c0ea98",
          "samples": 1052,
          "val_mae": 0.12576012313365936
        }
      },
      {
        "epoch": 13,
        "global_step": 4382,
        "model_sha256": "d2f86bcc0b831de5dbe9ba5ddbf2b4a53cdf96f420f8eb622d375e024bd364ea",
        "optimizer_executions": 4382,
        "train_batches": 313,
        "train_loss": 0.3254269192423728,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "c1b8f67fafa7b5b3d1defab13a79d52992483e09b5a202751e68f1e2e5a0e0b2",
          "prediction_sha256": "763399dbc3e4937d910149b473692eca84f12f962c69fcc78a9818c70799c0b3",
          "samples": 1052,
          "val_mae": 0.12903499603271484
        }
      },
      {
        "epoch": 14,
        "global_step": 4695,
        "model_sha256": "c25c254943a91895df1ac1bbbc89045081462428dae2a7afaf93de81209666db",
        "optimizer_executions": 4695,
        "train_batches": 313,
        "train_loss": 0.3206810803437817,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "996d46f09fe2a1d97f059643b74d126c48266049ea32d35fa5610e26fcc3acd2",
          "prediction_sha256": "19a949fca7359262c5ea077c3a1f489a15be9ebcfdf36fed56b5a890952b9a9d",
          "samples": 1052,
          "val_mae": 0.12585796415805817
        }
      },
      {
        "epoch": 15,
        "global_step": 5008,
        "model_sha256": "55697f6a7dbe8c77a960fe4f1d3df06c998081f7bf98334c9a1c7c5da3cb44c9",
        "optimizer_executions": 5008,
        "train_batches": 313,
        "train_loss": 0.31727013864936265,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "038780846792fbf89f1a8c2cd66e600b5c8018011969da4b1f52dd06f58e81fb",
          "prediction_sha256": "c721cec112931e52fe8d36141d22ad32eb33eac6b3efa3cbeb67ef8b0bacc8b0",
          "samples": 1052,
          "val_mae": 0.12256629765033722
        }
      },
      {
        "epoch": 16,
        "global_step": 5321,
        "model_sha256": "3fcb72ee1e82f0fab237d6d1f0eb209a7197fc33b852a11fd323e863e645a809",
        "optimizer_executions": 5321,
        "train_batches": 313,
        "train_loss": 0.3142588340808663,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "a21e528d0070916a6e7deb0a7d708cbe4d71f10f065d3f2699f84790d2e797ce",
          "prediction_sha256": "c27451bc37728aab3c9889ef5a2b86e53afa6b4c638b908b9eb12f5344fb1da4",
          "samples": 1052,
          "val_mae": 0.12310744822025299
        }
      },
      {
        "epoch": 17,
        "global_step": 5634,
        "model_sha256": "3920128ae31ed7b406e50b5fe2b20910b9ced728e3fe079f37fb904b8f8fc5aa",
        "optimizer_executions": 5634,
        "train_batches": 313,
        "train_loss": 0.3131025074977426,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "e08e7af9e61470f5aee9da3e6685def4409904bcaa3e8b911f92e6ac4c3e917b",
          "prediction_sha256": "502beafe86e63692dd9975569e6ef0bc921e1c5b2dd050fc00a4e6ade8ae2707",
          "samples": 1052,
          "val_mae": 0.12358620017766953
        }
      },
      {
        "epoch": 18,
        "global_step": 5947,
        "model_sha256": "e410d5175e0c215987f2f83f4e5aa49008a1c04cebe52115181327eb454955b8",
        "optimizer_executions": 5947,
        "train_batches": 313,
        "train_loss": 0.31095124810948377,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "318a9a467b4452879f8ea1e35dcf2315ac2126c5e0a8f88a659f5884f6eac49a",
          "prediction_sha256": "e6942083415c9f69c6f1bc087c47ef34516fb6c66889744db499ba8bb3cd21e2",
          "samples": 1052,
          "val_mae": 0.12029492855072021
        }
      },
      {
        "epoch": 19,
        "global_step": 6260,
        "model_sha256": "2ebdcb44ef6f4451f34bbed161543790ae9e568fe46fb6ecd6a499e9c955f162",
        "optimizer_executions": 6260,
        "train_batches": 313,
        "train_loss": 0.30927203995181174,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "7609f3d7d98162b58d1d807f45e066a55b56f9cd35568f3661486e6d4516ef88",
          "prediction_sha256": "eb797b7e3e9165792db7d8c8f2128a4e79b440f1f5153c6a78cf2ab924cc7e0a",
          "samples": 1052,
          "val_mae": 0.1254158467054367
        }
      },
      {
        "epoch": 20,
        "global_step": 6573,
        "model_sha256": "081b69e0375a770f7aa3d77d9df2c57342ba3505634651993739e1a9910534ae",
        "optimizer_executions": 6573,
        "train_batches": 313,
        "train_loss": 0.304572820510791,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "981a747d1f9c4d4877b28068240738890a71fa02eb5ba83cdb1b0022221db9d5",
          "prediction_sha256": "f0cfa7d81cfbff4c35257557f42e8da12f06e7f5cfad26c44829d795f75442a2",
          "samples": 1052,
          "val_mae": 0.12763629853725433
        }
      },
      {
        "epoch": 21,
        "global_step": 6886,
        "model_sha256": "a817d4122d4d6a6de0984ca8fa70230e4ffd3708d1f525c80a9d3159b8254576",
        "optimizer_executions": 6886,
        "train_batches": 313,
        "train_loss": 0.3034352069523538,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "0d097230962fb359ea7a036cae226d13b1f647af1888d24370b501fca3697238",
          "prediction_sha256": "80b394a2437d53560732c1693641ef7da3871cab8d6298c9c7fc3eba245314c5",
          "samples": 1052,
          "val_mae": 0.12381421029567719
        }
      },
      {
        "epoch": 22,
        "global_step": 7199,
        "model_sha256": "03d6429dd93e07b0053f3979d5c14d5be8f96096c15c84f05f62c0abea3e340d",
        "optimizer_executions": 7199,
        "train_batches": 313,
        "train_loss": 0.3012589024881067,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "5f3452fbc9e12be8d9036b1daeec263eca4c678de6f768b4e0b863585e21fc8c",
          "prediction_sha256": "80f61c7dbe7786a62243c98761e0a750b9576be2e9f7322c5cbeec0d62771f53",
          "samples": 1052,
          "val_mae": 0.12100210785865784
        }
      },
      {
        "epoch": 23,
        "global_step": 7512,
        "model_sha256": "1924b73a312e20d419fca37b68ea40676bfe4c7897516ebf758a92bfd153964b",
        "optimizer_executions": 7512,
        "train_batches": 313,
        "train_loss": 0.29807285991920285,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "74e4001154f95ca949cee510f973051a572f422d1e9dd0453a93a53dd32259d2",
          "prediction_sha256": "28c38a91af0bc0fd63169279b4210694e25f71ff392a44aca034d3e501b25ed8",
          "samples": 1052,
          "val_mae": 0.12207264453172684
        }
      },
      {
        "epoch": 24,
        "global_step": 7825,
        "model_sha256": "481b56cf251cbc73b30234979ea0349e4c43685f8baf429cf390ff5628ac9181",
        "optimizer_executions": 7825,
        "train_batches": 313,
        "train_loss": 0.2979048452549826,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "9959dc48ad8bdaed5c36539c4f9963a160b4fe8859d36cfb060f1381621b412f",
          "prediction_sha256": "c753699c1f2080f6ef6541e2dfc74e0df0c13d3dcfdf2a5cec0944c88d629551",
          "samples": 1052,
          "val_mae": 0.12007474899291992
        }
      },
      {
        "epoch": 25,
        "global_step": 8138,
        "model_sha256": "11441512acdc05a5f0671aea11af47b9a784cae1c5d5d05928b95ffb3f1ca090",
        "optimizer_executions": 8138,
        "train_batches": 313,
        "train_loss": 0.2947583878128094,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "1172b2ae86d5c111f3eb6d2d5176695059ff2bc5e7c9f2f7988dd338c4f6b521",
          "prediction_sha256": "bf3176d300d13d7442dd64ef951fdd983abc6545ca634baf09d02c0a79433321",
          "samples": 1052,
          "val_mae": 0.12018700689077377
        }
      },
      {
        "epoch": 26,
        "global_step": 8451,
        "model_sha256": "4bb7cc1fef8258eed0343fc34b370ffb0ebe3661a36c4586d9a2c6c76effd556",
        "optimizer_executions": 8451,
        "train_batches": 313,
        "train_loss": 0.2943087625555063,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "08be56b428965de2c4ff91d6979fc934bbe17963c268396a6b8a2c78dc91a30a",
          "prediction_sha256": "19f25a9ef12c32eff516f39bd709c8dfa4c1a2b4c7180c5533ff6d76086044fb",
          "samples": 1052,
          "val_mae": 0.12152110040187836
        }
      },
      {
        "epoch": 27,
        "global_step": 8764,
        "model_sha256": "e21a9197c5a08ca8c244b7de9f60a9ae73460ac5c67dcf6bf3d3c7b6e9374999",
        "optimizer_executions": 8764,
        "train_batches": 313,
        "train_loss": 0.2916749743570304,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "20b6f59e2fa39fa116874357ae4dd1617c72360586f1eea1445e80cf822c1bd5",
          "prediction_sha256": "58a7725f807b8abd5cdf0b3ed633e69cea2c0dabd518e9ff499a49cd00229260",
          "samples": 1052,
          "val_mae": 0.12159775197505951
        }
      },
      {
        "epoch": 28,
        "global_step": 9077,
        "model_sha256": "25ebd9276cd6b4c03cd916d6d86109807416d244c41f5c77cf5752a78c92c669",
        "optimizer_executions": 9077,
        "train_batches": 313,
        "train_loss": 0.2901848375091214,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "e08ed4d08c08c70005ff6bcb1febf0ab25cfec2261502a6d006ed9c99e860ce2",
          "prediction_sha256": "5ac0071281cedbbe04ce8d06c6475f43d67580fd8363930df40551a7e69d4c7c",
          "samples": 1052,
          "val_mae": 0.11949992924928665
        }
      },
      {
        "epoch": 29,
        "global_step": 9390,
        "model_sha256": "90848ac0d8eb1c4a4aa11524eb7d83ca8de8303255b6054577438aa4b82d6333",
        "optimizer_executions": 9390,
        "train_batches": 313,
        "train_loss": 0.289187669463018,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "4c9ca5a4312904809c217e7e536979a4d06be266d7466389ebaeab916a3c59b6",
          "prediction_sha256": "e32320c1af6d6c2463b7e625bbd4edf84d6cccbb89066cd03abc985395138956",
          "samples": 1052,
          "val_mae": 0.12046489119529724
        }
      },
      {
        "epoch": 30,
        "global_step": 9703,
        "model_sha256": "fc50b78ca4b6653b2b42296ad68315a4cfcb7679661cb08e8966cf70dc248baf",
        "optimizer_executions": 9703,
        "train_batches": 313,
        "train_loss": 0.28781855954577373,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "2d1b5d9274e1a1f4e4c60e151ea2744100896dcf3ec75fbaaa4878dd12a678d9",
          "prediction_sha256": "e245e2dd3d0a7d04cb25a00022ec69bbefef7c95cc0da8e288e360201a3cb067",
          "samples": 1052,
          "val_mae": 0.12044540047645569
        }
      },
      {
        "epoch": 31,
        "global_step": 10016,
        "model_sha256": "dfbeebeb380042a0fc96b3476b65473d4c688cc300ca77d10641cf9cde588ab9",
        "optimizer_executions": 10016,
        "train_batches": 313,
        "train_loss": 0.2857419111374876,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "622c57c58a0a4bbe2365a7bc03956a8b46c8d938d8bdbbee55b8ab0da900aeca",
          "prediction_sha256": "8226ed0d58c8588edbb9d997f946b5549430851a3cf4763ee4fe1a22e67abbf6",
          "samples": 1052,
          "val_mae": 0.1199253648519516
        }
      },
      {
        "epoch": 32,
        "global_step": 10329,
        "model_sha256": "905d48a9492196be0426c5370ffd1c832b4f57411931a4f873cb548368ca2275",
        "optimizer_executions": 10329,
        "train_batches": 313,
        "train_loss": 0.2835279706718255,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "2102602713e5e00c0f0866a1432be8e9ee7b6f5ddf53196bedf2f67276763116",
          "prediction_sha256": "82358b2e361619ceb23ad73e7239b37c0d286740f2be06ebae0b91fff622a905",
          "samples": 1052,
          "val_mae": 0.11972708255052567
        }
      },
      {
        "epoch": 33,
        "global_step": 10642,
        "model_sha256": "e11db9d20ff59b904e11b31f0f0bdcc67f43f72e8f7612032cd615509534b05d",
        "optimizer_executions": 10642,
        "train_batches": 313,
        "train_loss": 0.2822933342049594,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "6580f35db31f2cdc17a7ef414d77c1d6aa4e69bf38547ed9aaabcb4ba5d77a4b",
          "prediction_sha256": "000f926e67fc35add46612cadfa73abccddeb27bd09589b46ec3c2bfb312a90a",
          "samples": 1052,
          "val_mae": 0.11728474497795105
        }
      },
      {
        "epoch": 34,
        "global_step": 10955,
        "model_sha256": "d95564100e8122b8e125fea724e21c8cdb7dfaf8e4c1ed4ae051dbdf10320f65",
        "optimizer_executions": 10955,
        "train_batches": 313,
        "train_loss": 0.2794372228664342,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "65ff7b515bd73a70cd2aafb931cc13eb5526f892de589afcafaf9d7195103ae9",
          "prediction_sha256": "c4eefcd52c55bb99725c0d908b6e2ca597b7b2755d5603900e84af571bbe1388",
          "samples": 1052,
          "val_mae": 0.11931604146957397
        }
      },
      {
        "epoch": 35,
        "global_step": 11268,
        "model_sha256": "f385cbc2dd93e2ccae91994390532ace973fcc58b8e27fe4c59df45ee5503251",
        "optimizer_executions": 11268,
        "train_batches": 313,
        "train_loss": 0.2809130589265871,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "6dc65d8131025ca815209dc8ef3f2f5a4fc250765c6d5bb68e6ba0bb4ea2de4f",
          "prediction_sha256": "83f606c62e33a27e60fc43fc27c33b70ae79a03aadf82fdaaa04c81d122cec9f",
          "samples": 1052,
          "val_mae": 0.11850972473621368
        }
      },
      {
        "epoch": 36,
        "global_step": 11581,
        "model_sha256": "973031b0a77a9e94a660853980bbdd432a6ca9f4fb32ddb623f856924f16fc8f",
        "optimizer_executions": 11581,
        "train_batches": 313,
        "train_loss": 0.2781287076588877,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "362ae348f1f4431592b053a9447d321235727fab366ce1928e144eedc0fa28bb",
          "prediction_sha256": "a54555ecc4f3d100d44baa2e723f61c2879d5c41f91ef4ee124560abc18931d8",
          "samples": 1052,
          "val_mae": 0.11732521653175354
        }
      },
      {
        "epoch": 37,
        "global_step": 11894,
        "model_sha256": "f0f6e6ef5f75bdaba18dfecde3c493d359748a19ff444c80471a3304087be915",
        "optimizer_executions": 11894,
        "train_batches": 313,
        "train_loss": 0.2766272777821782,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "5244208a636fd36c70b10b80696d5878bd653ee25e35a61440d376519ef06a41",
          "prediction_sha256": "602eced2d285aa72d5010d460185d0c351cd4bf846422fa16bfe4a4ab40582ab",
          "samples": 1052,
          "val_mae": 0.11972317099571228
        }
      },
      {
        "epoch": 38,
        "global_step": 12207,
        "model_sha256": "ff89419c7472087402c089d4e84862a43d43fc7a94787bb230e92c54d7f9d256",
        "optimizer_executions": 12207,
        "train_batches": 313,
        "train_loss": 0.274906462777304,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "5d7312522798f3eea4fd522b8dfe6e07c03950149e60a1fc40c2faa373ceea97",
          "prediction_sha256": "f16a251d00fe529267ca24e1aa1cfc308e18fe7793c2767487f54be976f1042e",
          "samples": 1052,
          "val_mae": 0.11824418604373932
        }
      },
      {
        "epoch": 39,
        "global_step": 12520,
        "model_sha256": "b39012c8a02fa47fdec9867e24009f8f4d315e4e40a385d1a5d912a01c705b4c",
        "optimizer_executions": 12520,
        "train_batches": 313,
        "train_loss": 0.2742059660545631,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "bb586b227ba19fb61124d86895924e90570ca7d9ecfdcb7f7ef78fb365a6743c",
          "prediction_sha256": "cfda39db30ecc841cf5306b51e7fd64f3790af3e783364d67b07187ce90fe444",
          "samples": 1052,
          "val_mae": 0.12083075195550919
        }
      },
      {
        "epoch": 40,
        "global_step": 12833,
        "model_sha256": "d3165a9c53333e233155c7e19b21604d5b267455864de8d31fa39760ff93373d",
        "optimizer_executions": 12833,
        "train_batches": 313,
        "train_loss": 0.27368748502405393,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "65a1c66dc556659dadb7ded5e255e27dd88bbf1365f2f57bff5b00e369a32ae5",
          "prediction_sha256": "b933001bb390da994780c442c062e160ee22638617487f51ca961c0c7cd8ef2a",
          "samples": 1052,
          "val_mae": 0.11756183952093124
        }
      },
      {
        "epoch": 41,
        "global_step": 13146,
        "model_sha256": "197c31ef0d2e185debf534a02f12e5140c94c0fad1bb28f3e10a1fdf2a11ba0f",
        "optimizer_executions": 13146,
        "train_batches": 313,
        "train_loss": 0.27219590741728944,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "2591a00842e7cd748414f97fcf3aefa6260762ed3c6dee585b3ad9b6552dfa2d",
          "prediction_sha256": "4cc46f93f0006ed4394cc28069b5c39a91eeeb569c911d01d86bbc080757ff63",
          "samples": 1052,
          "val_mae": 0.11748319119215012
        }
      },
      {
        "epoch": 42,
        "global_step": 13459,
        "model_sha256": "6f7e89ac2eccdff9ab96f52628138d3e8b9904617238173860e62e18d751db9c",
        "optimizer_executions": 13459,
        "train_batches": 313,
        "train_loss": 0.2709449559893821,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "0da949244da4acdd4636c15d34fadfa74784b646ecc430b36469977e76a3eded",
          "prediction_sha256": "d359d6ed732d4ebd8c23feda777d5a59dd6d08e3c0abccb06aa2eca8823e4897",
          "samples": 1052,
          "val_mae": 0.1177375540137291
        }
      },
      {
        "epoch": 43,
        "global_step": 13772,
        "model_sha256": "f5e4617632f9643a784d15df49f2d014359ad84b622b0d341c2c604651439814",
        "optimizer_executions": 13772,
        "train_batches": 313,
        "train_loss": 0.27003243123319065,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "1139d634760fba54f23088e378140e2de4c209f6c4f8b974cd0626c393ae674f",
          "prediction_sha256": "bf05376ad483ac298d09f677d98c6d250b079feb386f468ca33b89694eac2cc3",
          "samples": 1052,
          "val_mae": 0.1193387433886528
        }
      },
      {
        "epoch": 44,
        "global_step": 14085,
        "model_sha256": "d4db4d88212d0286674914288b22b853f441b217603b7ca622bb6b35b9f21e2d",
        "optimizer_executions": 14085,
        "train_batches": 313,
        "train_loss": 0.26898533665448815,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "c142e10355627fb7b5d9fd08d994bfa23b71a8149674c8a289d84e81c7ae7c5c",
          "prediction_sha256": "6b0ca1db26a1ef6a58de7581b4da93dc68c0597bbc17e534eb03d4c95dd822d4",
          "samples": 1052,
          "val_mae": 0.11648917198181152
        }
      },
      {
        "epoch": 45,
        "global_step": 14398,
        "model_sha256": "03f6ccbe90f4f3f755399c6707feb210498a8224ad6b59eabaa69f81db474420",
        "optimizer_executions": 14398,
        "train_batches": 313,
        "train_loss": 0.2681978520573054,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "ef170a9abd2cef2913b60ad0189180417cf7c43ccaaf17532171b42d1dd59d00",
          "prediction_sha256": "314e33dfdc46b7a024fa5c9a81903a5304b48bb682586f4bc3b8a7ee8c510a47",
          "samples": 1052,
          "val_mae": 0.11781207472085953
        }
      },
      {
        "epoch": 46,
        "global_step": 14711,
        "model_sha256": "eb8bc208da39ec783120a988ae39df7823a6527a14f5e28cbeb9b19e870c6bf8",
        "optimizer_executions": 14711,
        "train_batches": 313,
        "train_loss": 0.26667994618587576,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "1f9e3393816e6cf18944689acc33f35c61b8d96c196abfd68c1407665c999818",
          "prediction_sha256": "07e87ffad0ec6e5a58baf5b91fc7cdf1d25832c49cc6823d2fc7bf3f725d7d00",
          "samples": 1052,
          "val_mae": 0.11756213754415512
        }
      },
      {
        "epoch": 47,
        "global_step": 15024,
        "model_sha256": "0147dd0e681b0fb61c6ad3bc653e379c7f876b48ad1a1046d2f2c497899e6d3e",
        "optimizer_executions": 15024,
        "train_batches": 313,
        "train_loss": 0.2641086805070975,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "6bb789a8c5b28e0c83a73ad2d595e7fcfdb36f55b005e49a8fc1d61e92fc598c",
          "prediction_sha256": "e2d5e23f0b04b4157c16c09a3625d3514470ef986a68e79a29722f1dbb2b6c96",
          "samples": 1052,
          "val_mae": 0.11639797687530518
        }
      },
      {
        "epoch": 48,
        "global_step": 15337,
        "model_sha256": "e6c864cc8eacd43951ccf4c126c060d1e4e206b3fd565595a9e2fef94e755a0c",
        "optimizer_executions": 15337,
        "train_batches": 313,
        "train_loss": 0.26542585169351135,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "304f0b7f52da5734f6d67a8e45f3bcee147401bf9c19fababdef6c12ddf6dd06",
          "prediction_sha256": "203faa8686a4ff25ca03c445c2ecf38e7d380f2b2b0c6796e3ea5ae5c7217672",
          "samples": 1052,
          "val_mae": 0.11780382692813873
        }
      },
      {
        "epoch": 49,
        "global_step": 15650,
        "model_sha256": "12da735d2442c29a5f2be56627e6ebb78b0826204c13ae695afa669c5a3ece3c",
        "optimizer_executions": 15650,
        "train_batches": 313,
        "train_loss": 0.26278929809045465,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "6968e60e57cd7e3a76d2ca77ef23383559c5cf9b777207244772c9db2b4ce10c",
          "prediction_sha256": "651f5093cdcf48ee2ebc9ee19ba0d6c7b41cd9e65dc6966033ffbfc72aab26c8",
          "samples": 1052,
          "val_mae": 0.11529969424009323
        }
      },
      {
        "epoch": 50,
        "global_step": 15963,
        "model_sha256": "dd6b08f678e0c438d15f1a30783820c0c631fd4ff0922406e48da7f6030f8ca0",
        "optimizer_executions": 15963,
        "train_batches": 313,
        "train_loss": 0.26187834596851456,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "2ea5cb8c479100cbc1e99aeb62270690ad1d6aea2b523a0abe5954d334788edf",
          "prediction_sha256": "70d3a8a4db570ac121741adfc5ec44619838155b08ed2e2c7893a17b76f6c0b8",
          "samples": 1052,
          "val_mae": 0.11756081134080887
        }
      },
      {
        "epoch": 51,
        "global_step": 16276,
        "model_sha256": "00c8e5815c3a06118ca494aa8015caff366f78f130ffaf70e2802ea6b14ea3de",
        "optimizer_executions": 16276,
        "train_batches": 313,
        "train_loss": 0.2594751356763231,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "bedd34cb9e633a7560badf670cd4bb03f5e9544db001772b402889fc216a72d6",
          "prediction_sha256": "846d25f4f56de4fe6d6848f69a33ca02fc635ca32648decb2f95a9100ef2b0ae",
          "samples": 1052,
          "val_mae": 0.113758884370327
        }
      },
      {
        "epoch": 52,
        "global_step": 16589,
        "model_sha256": "c9c93b5dce21e910eab3c08c84fcc094c2a1c86572c827c717ea51667d189e2c",
        "optimizer_executions": 16589,
        "train_batches": 313,
        "train_loss": 0.25876912695036824,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "b6d392cfd12b0d40e381900c3d83bf0cbf030241cb9043b8d096ffa30a5ec6e5",
          "prediction_sha256": "e07f9d5cd806262caaa1a2233ebc5b0007a0e7b4eabce1a732e10b7c34b23092",
          "samples": 1052,
          "val_mae": 0.11772795766592026
        }
      },
      {
        "epoch": 53,
        "global_step": 16902,
        "model_sha256": "fde8ee35a05dd6e32266e844499cf3a2abf990524ce95816012b6138885a1431",
        "optimizer_executions": 16902,
        "train_batches": 313,
        "train_loss": 0.2577918640934068,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "72416d11bc02ffe508c411b4370925e1edea33c8480486f3a5570be4d976e189",
          "prediction_sha256": "8d91f0b45d0daf174ebbe6d9ade935539cf648a706413774ddb4ecdbf7f56aca",
          "samples": 1052,
          "val_mae": 0.11541178077459335
        }
      },
      {
        "epoch": 54,
        "global_step": 17215,
        "model_sha256": "b6f44eeee17ab3692f063aaadf357e261b4aa59ad93fcc8cec6f93e04c1c2ba5",
        "optimizer_executions": 17215,
        "train_batches": 313,
        "train_loss": 0.25611492034929817,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "28d7c1f72db52d5951ad0bb76b4efb5391e5bbc5eb172c1d34af7deca77d1a4a",
          "prediction_sha256": "dd793fe4bf45ab475b92fcc7ecb199f6c8f60162c527ae20253a2c98ffea6c6e",
          "samples": 1052,
          "val_mae": 0.11886433511972427
        }
      },
      {
        "epoch": 55,
        "global_step": 17528,
        "model_sha256": "53caeb05a4536952954c3f2ffcf6034eab2b30aa755edf7a3668d71b892b0772",
        "optimizer_executions": 17528,
        "train_batches": 313,
        "train_loss": 0.25616138375853087,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "48ba2b7c026fc33957ab260bde4ede7966c2974439c8118ad71d654962de0ee2",
          "prediction_sha256": "7bb70be007642e533738d01ec37a4ccb24fa51f8c3ba2b291dccd80d032a9ac9",
          "samples": 1052,
          "val_mae": 0.11501162499189377
        }
      },
      {
        "epoch": 56,
        "global_step": 17841,
        "model_sha256": "13dc5b901cbe4b7832e1f48f608a5ef1895cce5e1323372f940a88157e321a88",
        "optimizer_executions": 17841,
        "train_batches": 313,
        "train_loss": 0.2548667641235234,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "9936efe4dbc36c9800364b10b56f7ed584ef2b070505788783f7ea1d1f5431ff",
          "prediction_sha256": "7334238fc8d737af86b4706ea944ea809eb8fd99cee3a4a4d4aceafa875f5ae6",
          "samples": 1052,
          "val_mae": 0.11604762822389603
        }
      },
      {
        "epoch": 57,
        "global_step": 18154,
        "model_sha256": "abc7435b11e6493be510602a958169e5e82b1c7024dde22e8c449213de8bb4af",
        "optimizer_executions": 18154,
        "train_batches": 313,
        "train_loss": 0.2538650876210177,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "81aa3d22c34941f2da06b66cfccff4d462e651bbdc93e5797c77025a1db04999",
          "prediction_sha256": "53141eabe1d00fe3dd9728adf3065e8032320c051b014c9d9f3835bf2c2ddcab",
          "samples": 1052,
          "val_mae": 0.11537528038024902
        }
      },
      {
        "epoch": 58,
        "global_step": 18467,
        "model_sha256": "80512c07a8ec9f0c6a9c33ba1e4f853a25da4c8f5610cc1b683b2a6f59e8d781",
        "optimizer_executions": 18467,
        "train_batches": 313,
        "train_loss": 0.2520227308309191,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "91517778efce6af65e2e9279629541a7984f15ec77f91a919085460d8227c224",
          "prediction_sha256": "84bb5cd5e3efabb387ea5fc63bd22f593c21c22523059d14019236205face7d2",
          "samples": 1052,
          "val_mae": 0.11650089919567108
        }
      },
      {
        "epoch": 59,
        "global_step": 18780,
        "model_sha256": "b268eedd204d28b98548178461579a21a26b6b54a904f112c120abfa832a2980",
        "optimizer_executions": 18780,
        "train_batches": 313,
        "train_loss": 0.2511649930288224,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "f0a8741484041dad4afe41cad81635c11eac03b944e09c4c732cc5d46517b2a6",
          "prediction_sha256": "66e74dd702b0f8d1955eab4257d3d0d4de3f695bc5fd58cf68c5ef6efc6cd97d",
          "samples": 1052,
          "val_mae": 0.11481019854545593
        }
      },
      {
        "epoch": 60,
        "global_step": 19093,
        "model_sha256": "3bc2a09dc1f771d299882b07a94b70d2b92506924bc2ab2482ab4caf25692655",
        "optimizer_executions": 19093,
        "train_batches": 313,
        "train_loss": 0.24882976254482583,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "98e35490fc347b93d1d26321ef55109fb578403fa169c0df8db6e596408bab17",
          "prediction_sha256": "769505eca831f36ac4caea74e1d315b45fe0330934af66d76447561bf1322efd",
          "samples": 1052,
          "val_mae": 0.11477117985486984
        }
      },
      {
        "epoch": 61,
        "global_step": 19406,
        "model_sha256": "d72cd51d686b1b4aa92a0653ba07d90975bd452414cf91299d67346457a30827",
        "optimizer_executions": 19406,
        "train_batches": 313,
        "train_loss": 0.24831362181479252,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "ed5ce78298527acbb8027e39a9b2abf043fd3654999d34a98f4026d20d84a98c",
          "prediction_sha256": "3d6d1bbce98392aa2244ac6ba48365c3c66046fb780ee8962c0d1308a36b2a4e",
          "samples": 1052,
          "val_mae": 0.11419576406478882
        }
      },
      {
        "epoch": 62,
        "global_step": 19719,
        "model_sha256": "cb525fb25d9e8cad60eb1169809319d3d3e7fcf8a7b182c45bfd226bc0ea5b3f",
        "optimizer_executions": 19719,
        "train_batches": 313,
        "train_loss": 0.247744898968207,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "10e4ae19699afa41936d262420ded01420bd7a978aaf4349922679654fea8873",
          "prediction_sha256": "28a7a93ee5e7b2492b6f8cbc97fbffa22a7159dd8a31f13826e4bebe4a4b5337",
          "samples": 1052,
          "val_mae": 0.11809994280338287
        }
      },
      {
        "epoch": 63,
        "global_step": 20032,
        "model_sha256": "4c740dde67ff5e2f12ade2c6c55c319659be69bb298b1dd1af8d40712ea2a3f0",
        "optimizer_executions": 20032,
        "train_batches": 313,
        "train_loss": 0.2469775782683687,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "aa004b034bed824376e2d69ab04ee70b1c62bda122a74dc970b5bd290e6edcfa",
          "prediction_sha256": "a3f4b8a6331c0f450df8b19a5bd2ce138fe0652279854ecc08834ab38bda32c4",
          "samples": 1052,
          "val_mae": 0.11551324278116226
        }
      },
      {
        "epoch": 64,
        "global_step": 20345,
        "model_sha256": "fc762416106901445a245d113b2769c2ddc9e318d0385b4dbda25a15347e902b",
        "optimizer_executions": 20345,
        "train_batches": 313,
        "train_loss": 0.2447651584081236,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "10beb1995ed4906f19a89344ee1dbf6e283a946b6a2a297c550281e182b67b99",
          "prediction_sha256": "bf48807257ce3910b26bffe6823c0d5bdc3635b39c3ec0bc0720c5d53a11f95d",
          "samples": 1052,
          "val_mae": 0.11356157064437866
        }
      },
      {
        "epoch": 65,
        "global_step": 20658,
        "model_sha256": "43625f57d6830a02b8fd9183b2bedea9557e8b94111d9bc453847914e1f6ee34",
        "optimizer_executions": 20658,
        "train_batches": 313,
        "train_loss": 0.2428289172204567,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "94d80167346472f5042c43e36fd1617aef59ff7f0a3164c5c25b42a7f0d31184",
          "prediction_sha256": "4b4f7c19008deec04c8d57008ca4d11fb332b3845d32ce9e6464aff2986795de",
          "samples": 1052,
          "val_mae": 0.11351598054170609
        }
      },
      {
        "epoch": 66,
        "global_step": 20971,
        "model_sha256": "ec1d72007614cc95cc21c43f96fbac9c536cb84c2973a34326d1da7a660312c1",
        "optimizer_executions": 20971,
        "train_batches": 313,
        "train_loss": 0.24323740843889216,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "e914b0c8ed11cfa8461baa0f2a3ff9b1ca4d3061f16bed67e2554c1a3c87a720",
          "prediction_sha256": "28c59102790c10a44b11416164457c4bb06deb233272f1c3525a96a259a1314b",
          "samples": 1052,
          "val_mae": 0.11310546845197678
        }
      },
      {
        "epoch": 67,
        "global_step": 21284,
        "model_sha256": "96c738c77a75a6643408758e816c86e18250e0abb4060428f3a6fb38c01d8109",
        "optimizer_executions": 21284,
        "train_batches": 313,
        "train_loss": 0.24136577329292896,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "016194b86c49ee02636062a382046ae47cdb58251533aa596bdaed2f2328b28a",
          "prediction_sha256": "0a72d0a02b15f6625ed7db386856a8b900d4de9366a1dd74cfe8f464cf4597b5",
          "samples": 1052,
          "val_mae": 0.11540985107421875
        }
      },
      {
        "epoch": 68,
        "global_step": 21597,
        "model_sha256": "0b21a06ea3ac9471716ad4d0dafe0b07765b3f5961f8a443e2679e93fde0889b",
        "optimizer_executions": 21597,
        "train_batches": 313,
        "train_loss": 0.24082214265646734,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "2778a854114f17ed6697436e646dbd44bbb97e0fbe511f3994321186a6189623",
          "prediction_sha256": "5db27104dade6043808ea39ffa698b8b88d8f564577b0f25b4486469a8c8b1fa",
          "samples": 1052,
          "val_mae": 0.1139536201953888
        }
      },
      {
        "epoch": 69,
        "global_step": 21910,
        "model_sha256": "0bfc1e5099942c180460e70025be9ccbe79b03f5001249be56e2f00f73057aa2",
        "optimizer_executions": 21910,
        "train_batches": 313,
        "train_loss": 0.23916158098229526,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "ee9464b88d8ac42efd0c5f60daa5b2cd424d38ed36adcb15f6b2ff633e9dfae6",
          "prediction_sha256": "246a292b56b7b5a950938f97dbb0b0addc00f96c4d1f2c58c1acb09b21375c5c",
          "samples": 1052,
          "val_mae": 0.11371354758739471
        }
      },
      {
        "epoch": 70,
        "global_step": 22223,
        "model_sha256": "70625c505a838ba76d0e1d84cbc034f2f4d2299e7a1faa6cf037e30c98914546",
        "optimizer_executions": 22223,
        "train_batches": 313,
        "train_loss": 0.23832158722628666,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "fc645211bb94d208084d4ecf9c580333074c29da3ae75e62900283a46c9ae513",
          "prediction_sha256": "2222058e57621bb0e43911be89c143a75364d7103dd239620acbecdc9d4d99c7",
          "samples": 1052,
          "val_mae": 0.11366022378206253
        }
      },
      {
        "epoch": 71,
        "global_step": 22536,
        "model_sha256": "159d5452a1ce188aba23aebf9443b92aef75e46128c41e8f78d41a0a50c380ca",
        "optimizer_executions": 22536,
        "train_batches": 313,
        "train_loss": 0.236625398476257,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "6f9c91b7983a785f3e0b292f3fc91fd6e1c56f4a75d9313cae6c5daf4edaaffb",
          "prediction_sha256": "eb6463bf20504cd46ba861b54f0bea3f330ef3554485a970a305e8409802b8ba",
          "samples": 1052,
          "val_mae": 0.11367756873369217
        }
      },
      {
        "epoch": 72,
        "global_step": 22849,
        "model_sha256": "f4115f8e2dd4e618a77b9b6445be818e7c805feb48f250c20a039c4aec1ee001",
        "optimizer_executions": 22849,
        "train_batches": 313,
        "train_loss": 0.23574089872297027,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "0d9da4803445c595f7b24b76faf2fb073724aafaca8a63b9e38e4e2f9ce68547",
          "prediction_sha256": "f705dc78484919958d37881a82c1e5c92fec36f754d24ac92032436ba5c89471",
          "samples": 1052,
          "val_mae": 0.11400224268436432
        }
      },
      {
        "epoch": 73,
        "global_step": 23162,
        "model_sha256": "7931295ee653967e1a68b9c695cc36519573ca9fd48087f2e99cdbef93f9ae4d",
        "optimizer_executions": 23162,
        "train_batches": 313,
        "train_loss": 0.23454170570442995,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "9edb31a9eb1e0c7b2c0bd2277dc92f4c4116a9ae9ed5e8a7068b94c2d234359b",
          "prediction_sha256": "75d07b03adb56fcc2bec345be65f6f605a59db0bcaaedd801b7ded3d9a9683bf",
          "samples": 1052,
          "val_mae": 0.11207252740859985
        }
      },
      {
        "epoch": 74,
        "global_step": 23475,
        "model_sha256": "61654ae3a380029b073b3b85703c0aabdff079af2b84a9310a5785d5c116ff28",
        "optimizer_executions": 23475,
        "train_batches": 313,
        "train_loss": 0.23267354161305792,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "915d9796d233bbbdbbc0577872e5b4edd96afe648dd97114da7be7a72f6c30cf",
          "prediction_sha256": "1327b107b9b2518b7c710f9f985710ef3ed0745797045cb76a52a57d58a4f838",
          "samples": 1052,
          "val_mae": 0.11091294884681702
        }
      },
      {
        "epoch": 75,
        "global_step": 23788,
        "model_sha256": "39c96704bd37ae1ad3b012db23848256804cd9d1fb71db71f9a6d4179ed84a29",
        "optimizer_executions": 23788,
        "train_batches": 313,
        "train_loss": 0.23294404409533598,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "b5e3d6565b79294a97e40b9018e4104b3a7fa642556401034f947af033ff41be",
          "prediction_sha256": "0855e9d83b858e983525618958cf691ff097cc67b565bd367009a60a27eec80b",
          "samples": 1052,
          "val_mae": 0.11569477617740631
        }
      },
      {
        "epoch": 76,
        "global_step": 24101,
        "model_sha256": "f131ff9890018122b41ec915f374327df8df4542086e2a553702cc63892f947a",
        "optimizer_executions": 24101,
        "train_batches": 313,
        "train_loss": 0.23072754436537649,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "4571513e5c462fce7cd53fb360dcec7a252d4c239c651fc51467cb28c3646718",
          "prediction_sha256": "385256bf6948834153f21eaac7cb41d221282b86d4d29336618fb757c0d61bd8",
          "samples": 1052,
          "val_mae": 0.11291209608316422
        }
      },
      {
        "epoch": 77,
        "global_step": 24414,
        "model_sha256": "38e2ed4c986b1d07abea67c1a90062e8ab2b1a1a3816ea01ff302c1aecc05776",
        "optimizer_executions": 24414,
        "train_batches": 313,
        "train_loss": 0.22946024261628492,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "47e8a5792ecea6b1a3ff2d7e50f3fd21e9745ef7fc34c829a45e1b21d068976c",
          "prediction_sha256": "bb3cd38baec4685393a5714a639a1c20aa47c454e7a425766dc99b9aaf0c8683",
          "samples": 1052,
          "val_mae": 0.11333758383989334
        }
      },
      {
        "epoch": 78,
        "global_step": 24727,
        "model_sha256": "a5f77ff36f97ae7b95f7ea947ef1fe66570cfce2b0a663b78f1f752f8d4638c1",
        "optimizer_executions": 24727,
        "train_batches": 313,
        "train_loss": 0.22850643975917523,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "23e187e5a43d737f4c78cffbae9bdcc875f7b007cb6117945bdaf13550c2cfc9",
          "prediction_sha256": "aac200157ae15ad33a2eb867f2d56b46e254fc506e23f487e828d1f205917cd1",
          "samples": 1052,
          "val_mae": 0.11129346489906311
        }
      },
      {
        "epoch": 79,
        "global_step": 25040,
        "model_sha256": "984b8ff83bca0098161c8511d88f1e3d8399462c77db2fd060ea6f31f8b8b474",
        "optimizer_executions": 25040,
        "train_batches": 313,
        "train_loss": 0.2268821030893344,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "36d34452346dc079a2ffa4b77e15ff10f2ebe75d9a0b747e3accd253001b9937",
          "prediction_sha256": "d447fdbbb5ca8433d6c97c646e11f2638f6663c07c475268da6a2be725516b1d",
          "samples": 1052,
          "val_mae": 0.11160298436880112
        }
      },
      {
        "epoch": 80,
        "global_step": 25353,
        "model_sha256": "0442348bad7d9bee7d5288a021759717ad45d5ccf4b55c8315cbba34f22e1905",
        "optimizer_executions": 25353,
        "train_batches": 313,
        "train_loss": 0.2258644558719507,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "e0ebc1ff00721519d8d9f4af7774a5ba17f1702c103c0d7f0305cb08559e0e40",
          "prediction_sha256": "72c4ff1f30af8634080e0a873938e5dfe14c1b7584bebb91d5fdffaadcef51f2",
          "samples": 1052,
          "val_mae": 0.11367233097553253
        }
      },
      {
        "epoch": 81,
        "global_step": 25666,
        "model_sha256": "5fdf0e351053df37479d0614f57daef8c31a22f9dabb25bd4907236a869dfd29",
        "optimizer_executions": 25666,
        "train_batches": 313,
        "train_loss": 0.2252686808414415,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "603bf021a803d4b7f8ec1785a9832aa7ade8674324a9206e11f38188fe9637e4",
          "prediction_sha256": "73690c864bc291a5f450111f92b08cd303c210f871e4d689fd368507840eaa5e",
          "samples": 1052,
          "val_mae": 0.11265619099140167
        }
      },
      {
        "epoch": 82,
        "global_step": 25979,
        "model_sha256": "f55ec62752e8bcdf0cefc66329b1957737da7ecd4c692e6593c26f5e6173ccef",
        "optimizer_executions": 25979,
        "train_batches": 313,
        "train_loss": 0.22372558322242111,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "3201eea0cad8efcda881ac162cfef15b02407c88903327cebf3e75b4413bde7c",
          "prediction_sha256": "07560be0d36022245751b6f0b010d9e91112a47d2c30df27dddf78a5289636b4",
          "samples": 1052,
          "val_mae": 0.11163987219333649
        }
      },
      {
        "epoch": 83,
        "global_step": 26292,
        "model_sha256": "65cddc1e5d3a2638f573f5fe0dc599bf025b61b2d5e9e0f682e4d9f18640bff5",
        "optimizer_executions": 26292,
        "train_batches": 313,
        "train_loss": 0.22390696927797743,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "7692d9ab26d0cc47398f2b5f423135f36070ae8ff36b325da43f430b2fd54a50",
          "prediction_sha256": "1a6b04a39c689c931cf7ddf4d1802fbd00cb11ac7125ba770dae3967ec271843",
          "samples": 1052,
          "val_mae": 0.11111963540315628
        }
      },
      {
        "epoch": 84,
        "global_step": 26605,
        "model_sha256": "7bb4ada8715b18c88aef81044d576b43c68b757b620025a3a1510f1c2b0f7a68",
        "optimizer_executions": 26605,
        "train_batches": 313,
        "train_loss": 0.22098784627190432,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "bd527ddbc0c12c2975d3ccc41fa455b87478a3339a0b803c1e854e54080b8a42",
          "prediction_sha256": "268fd0e03d69d259c8e6b3c4bc01729564835193516384a5670b0a90eed803cd",
          "samples": 1052,
          "val_mae": 0.11346808075904846
        }
      },
      {
        "epoch": 85,
        "global_step": 26918,
        "model_sha256": "1c2eb453c62daae6e17611c4c89f5937515808e54613a43f22b2ec41f8a0e138",
        "optimizer_executions": 26918,
        "train_batches": 313,
        "train_loss": 0.21976455241770904,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "5a096e3ea8549b25d8a83c87d51f5eac4368de34c7e9053f941efed791d762f9",
          "prediction_sha256": "6d4ecc5228dab51ba8078a02a3b5d15ddee46f4f8e477781d632c6fb30f8a2a6",
          "samples": 1052,
          "val_mae": 0.11129086464643478
        }
      },
      {
        "epoch": 86,
        "global_step": 27231,
        "model_sha256": "dc881250665854ff1ab94af377ee214a2bd7d2be57b7b5e63e157647c1f8ae9a",
        "optimizer_executions": 27231,
        "train_batches": 313,
        "train_loss": 0.21855159961396797,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "7366561da72ae6acaf99868bb0af66147ef7dcbc34fb60d74be821976bf92a6c",
          "prediction_sha256": "789cddd65f7717193796ffbe3f8dec609aaf7f0d5a4c6afd8237503c4433c062",
          "samples": 1052,
          "val_mae": 0.11159604042768478
        }
      },
      {
        "epoch": 87,
        "global_step": 27544,
        "model_sha256": "5e9e45eb0cd696500bfb9d51c0ff6fe5ea73e936e0bbf86abb8cdbaa15cd5f46",
        "optimizer_executions": 27544,
        "train_batches": 313,
        "train_loss": 0.21674968632618216,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "0bf55e557799212c26381c28e7c1d0536df801099f5277465acbb7fb624e90d4",
          "prediction_sha256": "bf6279dae98297970a86dea14a78a2be0614177ee43bfece8727604ecfc55c3f",
          "samples": 1052,
          "val_mae": 0.11060632020235062
        }
      },
      {
        "epoch": 88,
        "global_step": 27857,
        "model_sha256": "7c45534a1168d492ac6bf354dbfb279353d7aa2a19a4b5670918bc01fa8e3c15",
        "optimizer_executions": 27857,
        "train_batches": 313,
        "train_loss": 0.21560967089234112,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "734291b4d93b64c31a328c9ba7c04551012cfafaa82d49010a3105768e1cbd8e",
          "prediction_sha256": "59d17e020078b32a3f7065929ff6ba66b162c9533506a12052cb58a5093d1f7a",
          "samples": 1052,
          "val_mae": 0.11014391481876373
        }
      },
      {
        "epoch": 89,
        "global_step": 28170,
        "model_sha256": "f7f6e1ae6eb5b30a830516c9ce0e7c7510bdea877ab6389d51ed899b403f5552",
        "optimizer_executions": 28170,
        "train_batches": 313,
        "train_loss": 0.21498924685331733,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "d8207d47aedfeb373f8f3ec1852353ecadc4c29864fe044b484d1ea479d154ca",
          "prediction_sha256": "f293804a6cb2de15610d598ce742754f15c08b87166f9f911ba05f2a1916dd38",
          "samples": 1052,
          "val_mae": 0.11086487025022507
        }
      },
      {
        "epoch": 90,
        "global_step": 28483,
        "model_sha256": "a7df6f0b70f5cd5632083237029f8469ce26fc4fdbad06dde599fd6fc2e9f4f0",
        "optimizer_executions": 28483,
        "train_batches": 313,
        "train_loss": 0.21292988022594656,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "af1c3f2f50ff2b1a89fb0581340b3bc6fc3659a3441a553ec316e6d42d9eaafd",
          "prediction_sha256": "a413ffbe98813a23dfd64708e2493debc3de899a7bc2f8f929b777065e462ebe",
          "samples": 1052,
          "val_mae": 0.11158102005720139
        }
      },
      {
        "epoch": 91,
        "global_step": 28796,
        "model_sha256": "959fe30947ba161bad110a5644fde978b536b42a61607026c754908ea7b79221",
        "optimizer_executions": 28796,
        "train_batches": 313,
        "train_loss": 0.21231649553888374,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "7c3198f169b263f5ce720e2b67eb94b652a0682ed3fdb139e93ccc10ab3d1f88",
          "prediction_sha256": "5e9f5468856cd63c2bc50251c0fc71bb68a482783a6fc5f5dfb86a6ae27ccfb8",
          "samples": 1052,
          "val_mae": 0.1097688153386116
        }
      },
      {
        "epoch": 92,
        "global_step": 29109,
        "model_sha256": "7b3eca33c757ef3068be25ee0646691b4bfa67d7f387acf26612f31da9d77bd4",
        "optimizer_executions": 29109,
        "train_batches": 313,
        "train_loss": 0.21138670589707462,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "9d80167e44faa1371f0b2ff995b8e259fb21cdcc36056500346e8af690e132fe",
          "prediction_sha256": "500caa96be847ecd86ca0962b6a2b9b4218b536907ccde0ad8ce191f4ef5c0a7",
          "samples": 1052,
          "val_mae": 0.1118139699101448
        }
      },
      {
        "epoch": 93,
        "global_step": 29422,
        "model_sha256": "48442c4c04d6efcaf87ba6f0c211c4c9c6b539f211b647927357603b3dd2d862",
        "optimizer_executions": 29422,
        "train_batches": 313,
        "train_loss": 0.2098890093396869,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "d4237f677439879ba12e574f9bb2ab8603ba07879a82fbbce4a2d8797dcd47ec",
          "prediction_sha256": "a00b641c12a1e0f7ca325212a3732965a5346017c5d61bc8339ddf45cdeb8586",
          "samples": 1052,
          "val_mae": 0.10949408262968063
        }
      },
      {
        "epoch": 94,
        "global_step": 29735,
        "model_sha256": "37e8bebbc8edae3815ced779e6048f9d667c6676fc6913ed3f03fec46fc3306a",
        "optimizer_executions": 29735,
        "train_batches": 313,
        "train_loss": 0.2088601232691347,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "d57f7e488e8b32964e1a7ed79e21f905933991db08d10b42692e07fbaf609ba3",
          "prediction_sha256": "d017d3f724e548d1b3faee264bb0420df015809a6faf0b0d642e7ff9f5c0cc89",
          "samples": 1052,
          "val_mae": 0.10882927477359772
        }
      },
      {
        "epoch": 95,
        "global_step": 30048,
        "model_sha256": "34173c4a2441d5c59b649954283bf0c23d18b44295d8e01988e656da07ae12f3",
        "optimizer_executions": 30048,
        "train_batches": 313,
        "train_loss": 0.2069464523624912,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "3faa2feba0cd0a679b08410beea043efaa52b775b4ac98e5994aa981c3056cde",
          "prediction_sha256": "f339d4b95d03c47c20e7b4feff05509ffe3fc2247f38680f3e399a82c8bf5046",
          "samples": 1052,
          "val_mae": 0.1092953011393547
        }
      },
      {
        "epoch": 96,
        "global_step": 30361,
        "model_sha256": "9886b0c859ff0c4a02e030be04cc90ecdc5f7e41d57bfb86a42f71da6e4da0a1",
        "optimizer_executions": 30361,
        "train_batches": 313,
        "train_loss": 0.20586320343295422,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "b1656243bab426e375116757888b35fc6f23b2a7946ef227ac324ad7eefa89f0",
          "prediction_sha256": "eb9e248c80bef3bd2a789f33765e6b34ddee58a3abd240a76a8b3c233a37cea9",
          "samples": 1052,
          "val_mae": 0.10988498479127884
        }
      },
      {
        "epoch": 97,
        "global_step": 30674,
        "model_sha256": "221483f3b321fccf45e29376631f20e539f2c3623b38fc62cbfe1bfe987dc22b",
        "optimizer_executions": 30674,
        "train_batches": 313,
        "train_loss": 0.20398960169389838,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "1f9908dd396316a49dfaf8326e791a06d9d6c9324a57f504b7e88dbe903b085b",
          "prediction_sha256": "bcb5a443e614dfd31e7e22304bfc1be1a7e941a2ef413bf80035fa22d3871836",
          "samples": 1052,
          "val_mae": 0.10934130847454071
        }
      },
      {
        "epoch": 98,
        "global_step": 30987,
        "model_sha256": "cc0d13025362f1f10c961591f811483b8ab5f4cd46ac41b05eb2d33e227f81bb",
        "optimizer_executions": 30987,
        "train_batches": 313,
        "train_loss": 0.20385548331908498,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "3c2ca19c089b22a485ada9ee58624975749bca45e844ea2bceb7195db82eebec",
          "prediction_sha256": "037add58dc520104c18840ed20804f7bd9fe46311d37d13fd3fb800b2b6078e4",
          "samples": 1052,
          "val_mae": 0.10946108400821686
        }
      },
      {
        "epoch": 99,
        "global_step": 31300,
        "model_sha256": "13824ce83797f00f3d102930892572cddf24667cf70f21c25982817f9e7ec8bb",
        "optimizer_executions": 31300,
        "train_batches": 313,
        "train_loss": 0.20205400575012594,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "a485a91178df39623628129ffa730db4892f8473b212e31d41ea89c9b257c489",
          "prediction_sha256": "3398eca87ace846a985993844c164200e46df7f97316c98626d3ad767bc1b68a",
          "samples": 1052,
          "val_mae": 0.10951030999422073
        }
      },
      {
        "epoch": 100,
        "global_step": 31613,
        "model_sha256": "c439559d85c9f908a3879c466644543697e3872cdf81fd8b3c488a6a8f9f006e",
        "optimizer_executions": 31613,
        "train_batches": 313,
        "train_loss": 0.20091927254535835,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "ca4a2f5efb7dbb6f4bf639c8b19de0cdf468f9ca4fce422ebaf0f49ae8e8cef9",
          "prediction_sha256": "c56792d27486a4969be0d48c801f453dde2976727f068a29686e62715a49bbfd",
          "samples": 1052,
          "val_mae": 0.11093848198652267
        }
      },
      {
        "epoch": 101,
        "global_step": 31926,
        "model_sha256": "2fcf5aa93c268bba7958b1c0370d6458c540daccc4b3d967a05f1318ad661858",
        "optimizer_executions": 31926,
        "train_batches": 313,
        "train_loss": 0.19949101256536106,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "3e32ac925bd20f671f2b265f71d3d4e5a6d7b47e6c05b97d812a69a6d1dbfcd7",
          "prediction_sha256": "23be672085aac8f1a55037e0b9fc4157fc4ccc24e9bfcb37606a9c2c2cda476d",
          "samples": 1052,
          "val_mae": 0.11043553054332733
        }
      },
      {
        "epoch": 102,
        "global_step": 32239,
        "model_sha256": "5685b7c34c1139d06bbc6c5416a34c71658a7b88fc83038b553681a4315280b5",
        "optimizer_executions": 32239,
        "train_batches": 313,
        "train_loss": 0.19865427795115978,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "881d3662d897ecdd5b4e6de4fb4884ee603ec397f264ead78a9053a13dcc3389",
          "prediction_sha256": "123b2f0c2d32d3f035f9178397ac8beef83bb677e1e812b6051e72a778daeb3f",
          "samples": 1052,
          "val_mae": 0.10894479602575302
        }
      },
      {
        "epoch": 103,
        "global_step": 32552,
        "model_sha256": "5ad346904f7dd590746f26253331565ee3e141616bd71ba7ab019747e616ccca",
        "optimizer_executions": 32552,
        "train_batches": 313,
        "train_loss": 0.19718928242017883,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "cc2016e243e81338dd7e64d73e809fa22d6d16f502b5e59a5c3a58a1e4a16b95",
          "prediction_sha256": "f7d94665c2af49c68f9fafed901a77af7d7f6f028bfb7a9c8c81372ca38a64f3",
          "samples": 1052,
          "val_mae": 0.10863269120454788
        }
      },
      {
        "epoch": 104,
        "global_step": 32865,
        "model_sha256": "fef8525c162787bfb784c02dc2fc2241878b08cc1cede3b4692ec512c562f40d",
        "optimizer_executions": 32865,
        "train_batches": 313,
        "train_loss": 0.19630652335008011,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "3c33686b13f36de5ee626d46f702c28375f9a5084f22d80ba33108604127fe1a",
          "prediction_sha256": "90e71f902ca1979fda08798dff2e030ce51d2d11d239120a0c775d938a967115",
          "samples": 1052,
          "val_mae": 0.10880643129348755
        }
      },
      {
        "epoch": 105,
        "global_step": 33178,
        "model_sha256": "85d8a82e1832559fc6253a55229f9aab9c700022453efb7af7a3c312a01c56c2",
        "optimizer_executions": 33178,
        "train_batches": 313,
        "train_loss": 0.19533750918630183,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "7f64436ae0f33da4be391cb0c9853da25869433286b4452a5f42845fd7d5ac6b",
          "prediction_sha256": "8ef9849446b1e8f3f988edd35b1fc62122177983d0da6eb5366301cbb23dd6d7",
          "samples": 1052,
          "val_mae": 0.1104636862874031
        }
      },
      {
        "epoch": 106,
        "global_step": 33491,
        "model_sha256": "eb9a1f5d58ee1668638ecb6090c7e03350f615e6ff9d68eb5ee1c0f8bac4f0e3",
        "optimizer_executions": 33491,
        "train_batches": 313,
        "train_loss": 0.19335087356671193,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "e0cd36c34c0497b254b0c251481d54bb55522de2e9eea2b75e6bc96cb58b5a18",
          "prediction_sha256": "61e64c01e509aa94bca6f28ea5e0b398b34616a5c2155648c626a228861c98ce",
          "samples": 1052,
          "val_mae": 0.10869110375642776
        }
      },
      {
        "epoch": 107,
        "global_step": 33804,
        "model_sha256": "f242ebeb3cc87aa3ffca2b4d866c28a47df90df26f62276c818a3ad07a47745c",
        "optimizer_executions": 33804,
        "train_batches": 313,
        "train_loss": 0.19238009257174996,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "06ea3cb7c135dbfad03c2879e86a3b7bef2de9cd95138ddc0f4bc7daee10fa63",
          "prediction_sha256": "0c3634bd57f338c41a1b64be84ed6566ef16be258f7f887a5ce3350441ce0a95",
          "samples": 1052,
          "val_mae": 0.10933054983615875
        }
      },
      {
        "epoch": 108,
        "global_step": 34117,
        "model_sha256": "9d42c7ca9969f89785f6d836f97b74b5232f650978e244c590a92b59ed05547b",
        "optimizer_executions": 34117,
        "train_batches": 313,
        "train_loss": 0.1916383554840233,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "4df6d62424e3ecac6407384319580824f93026714f080254f2758318638fbf44",
          "prediction_sha256": "8728a15ae23975bd30520d0173a0215f2c079a4dd2fa38ec2ccb5f5347ee99e4",
          "samples": 1052,
          "val_mae": 0.10827038437128067
        }
      },
      {
        "epoch": 109,
        "global_step": 34430,
        "model_sha256": "591d2b30db821bccf65d460e37efaa7c178b5ffe8104f25e79bc259d0e6e5f26",
        "optimizer_executions": 34430,
        "train_batches": 313,
        "train_loss": 0.18855242852174664,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "a1bbb81eef631f8da7871e524864e0b090f53dc3ce83ffeeb9931ab7ad320f3d",
          "prediction_sha256": "4e60229a9b6034d7200899f1e671f7606afeb0269a4122c1e90e61d975e25677",
          "samples": 1052,
          "val_mae": 0.10671043395996094
        }
      },
      {
        "epoch": 110,
        "global_step": 34743,
        "model_sha256": "63fafdfa40ba05b598e5435f65ecb731ab68b9adea1a30442cc70af66674bdb0",
        "optimizer_executions": 34743,
        "train_batches": 313,
        "train_loss": 0.1882290890741791,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "6f77ff13fdf6866c1c944d484a52cefcf485ac6b8cf05d62bd18d6628fff4cf2",
          "prediction_sha256": "f114af0a9c5cdef4a601b6b9208e08dc5b569dfff92b41228ab48a5f76e4b4c1",
          "samples": 1052,
          "val_mae": 0.10888304561376572
        }
      },
      {
        "epoch": 111,
        "global_step": 35056,
        "model_sha256": "db0b38d25930831aa5e3128c66e9daa2623b00b9776c74abd846eded99c3753f",
        "optimizer_executions": 35056,
        "train_batches": 313,
        "train_loss": 0.18726339267648198,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "ff574ac294fb41224d6686673697014cd97dae7610b0a21e83ef971391af1eb7",
          "prediction_sha256": "7494894cdcb0b741a510376b273c9cff38ab7f6bc4665da7b7a17b3b2164a158",
          "samples": 1052,
          "val_mae": 0.1088591143488884
        }
      },
      {
        "epoch": 112,
        "global_step": 35369,
        "model_sha256": "87c0e0dd8a9f76bc4fdf7c1e892703eb8d33449823edbb4ff74717d91cdda98d",
        "optimizer_executions": 35369,
        "train_batches": 313,
        "train_loss": 0.1858386378326816,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "e19c6ea9385b655f5257a2d7121c79f1c9cc2afbd0bf2f164b4078edd9a4d45f",
          "prediction_sha256": "f67e062a30ae109560558ebd1e22536a3a96682da37dbc720e3786dd3a6a252e",
          "samples": 1052,
          "val_mae": 0.10786525905132294
        }
      },
      {
        "epoch": 113,
        "global_step": 35682,
        "model_sha256": "ab91e098a878a28f10eb28131f7f4e9114db3a310b201d634dc0e893a21b2591",
        "optimizer_executions": 35682,
        "train_batches": 313,
        "train_loss": 0.18506464541521342,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "2519f772c885f4c5a4e6a478668cb5c1b0cc9608ff411c6a71cc34ce8b385da1",
          "prediction_sha256": "e5fa831b509ec66744a9534f16bb833d3a62770d5931ed549c3f700121df999e",
          "samples": 1052,
          "val_mae": 0.10697589069604874
        }
      },
      {
        "epoch": 114,
        "global_step": 35995,
        "model_sha256": "bdac7b2311c5502b4dd3bc73e2fd52036ee84c45623256eb436e555d39cf3126",
        "optimizer_executions": 35995,
        "train_batches": 313,
        "train_loss": 0.18422619474636415,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "4f267adbf4f3a554a8d4eb2df357b7ad68016e6d9c21083f8a23385ecca7646f",
          "prediction_sha256": "97d3dcded148d1164990d55d9647a1e4ab1ee3c71a62bbf8492268d3261eb842",
          "samples": 1052,
          "val_mae": 0.10710163414478302
        }
      },
      {
        "epoch": 115,
        "global_step": 36308,
        "model_sha256": "12d5efe65fdbf8a09c618f06601d8f2372b807d0f412c60874db57dcfaac2059",
        "optimizer_executions": 36308,
        "train_batches": 313,
        "train_loss": 0.18263184835104745,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "56653fa4e760b1faeed56aefb8dd733e325ea7e34c86db22c25d0946e5df724c",
          "prediction_sha256": "e384a8652cf6dbf2fde6355e465d85de8efe6dd4e50b68a9e5c52182c45dad1b",
          "samples": 1052,
          "val_mae": 0.10693521052598953
        }
      },
      {
        "epoch": 116,
        "global_step": 36621,
        "model_sha256": "0d8d9337e1bb16dd74ac9a3c6d57da3690baecb902f899ae218db3deed780ea9",
        "optimizer_executions": 36621,
        "train_batches": 313,
        "train_loss": 0.1811119759008372,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "dafab770eef98fa38e35c7724397711ba7270512542724f5c10034779c3f676f",
          "prediction_sha256": "9000d626e9f6d53ecdbb6eadb62086692c033cdd995e7380d4d3c814d1ab563a",
          "samples": 1052,
          "val_mae": 0.10758034139871597
        }
      },
      {
        "epoch": 117,
        "global_step": 36934,
        "model_sha256": "159ad13cb29dbe3d39a474cfcc6e39ba0be3c01d40d8e8213a51b6975ff0cf90",
        "optimizer_executions": 36934,
        "train_batches": 313,
        "train_loss": 0.18013951510109863,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "8f4874bb8329a0ee4be537a2a61b014a9a916cbb35ece6b1d923637d7722134d",
          "prediction_sha256": "9a7288828c8ab1cc5cc445b82c29f77ed0312988517ec84364918486b9f5955e",
          "samples": 1052,
          "val_mae": 0.10816651582717896
        }
      },
      {
        "epoch": 118,
        "global_step": 37247,
        "model_sha256": "b1abdfcf8f9bbb04a4696aae3791a0e785b96b06f1825bbe024901a2fa7a19a5",
        "optimizer_executions": 37247,
        "train_batches": 313,
        "train_loss": 0.17897634682644648,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "65eeae41cfcda12d236f534ea9c03c899a1671caf6c4bbaef1b8233817184715",
          "prediction_sha256": "8739f93eddde81025aedd6dd8b55b1d78a43e053c303f6481169a3098047e406",
          "samples": 1052,
          "val_mae": 0.10799803584814072
        }
      },
      {
        "epoch": 119,
        "global_step": 37560,
        "model_sha256": "9fe0de85c53426b34680a44febc25fd8b975fa1ba961e4c9e4e0114c55be0003",
        "optimizer_executions": 37560,
        "train_batches": 313,
        "train_loss": 0.1786671252403905,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "070d2a2e4512d816583f324fd83b4c4d9c5c85790bb366c576face135cf2a378",
          "prediction_sha256": "3a693bc9600d5dc5e331427dec1fe7f7df5f1e2ce04f0f5494c83ec13e09080f",
          "samples": 1052,
          "val_mae": 0.10802358388900757
        }
      },
      {
        "epoch": 120,
        "global_step": 37873,
        "model_sha256": "90eb5980126d4d8fdae568e6a80f70f175981361578b888b5a268c5532e0253f",
        "optimizer_executions": 37873,
        "train_batches": 313,
        "train_loss": 0.17703681690455894,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "084523f6ef763fe099973a6d9edea9ec874c708fdf16e59461cbc40f20767787",
          "prediction_sha256": "437482e2501a2f8e4299ecf9d01c1134ee43d92e4bbc3a6846ed5dff417dd3b5",
          "samples": 1052,
          "val_mae": 0.10719956457614899
        }
      },
      {
        "epoch": 121,
        "global_step": 38186,
        "model_sha256": "b2a3902e9a2c466832def4389409436fa842ee708c39432816831d321b51139f",
        "optimizer_executions": 38186,
        "train_batches": 313,
        "train_loss": 0.17628416494882754,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "74cd579a4cdc378444f0828cc2db16ecc0a77864df05bd83f39228f7fdf7de63",
          "prediction_sha256": "4ec66ec764793ed2816e72e94e9fcb446e8b690ec12372db58d7415b7686cb6f",
          "samples": 1052,
          "val_mae": 0.1077805757522583
        }
      },
      {
        "epoch": 122,
        "global_step": 38499,
        "model_sha256": "0afe5b7b084aa2a7951520edef7d3278d3aea955b8454bd4d60a0bfa5658adbc",
        "optimizer_executions": 38499,
        "train_batches": 313,
        "train_loss": 0.17545889325860942,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "64369f23c205f64b91da3704dae529e8edb6e263754a6fb655c284cafba8d52f",
          "prediction_sha256": "effe9275f1b8d7b20313c17c735941c7ec687ae48df179c0265c568a8a1dde1e",
          "samples": 1052,
          "val_mae": 0.10737982392311096
        }
      },
      {
        "epoch": 123,
        "global_step": 38812,
        "model_sha256": "4e17d59da0e62e34fb1caa710af23113f09bc9c062067f18557197748920b85e",
        "optimizer_executions": 38812,
        "train_batches": 313,
        "train_loss": 0.17415500076480803,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "2e37781c17c02d65df458f262ee398bb5ef93c7dec4c8b49e585422c257f0056",
          "prediction_sha256": "f91d1f8eb117feabe16ba3cfdbd78375d6727fcf534c4d9dc1a23fcb9c365fa7",
          "samples": 1052,
          "val_mae": 0.1072564348578453
        }
      },
      {
        "epoch": 124,
        "global_step": 39125,
        "model_sha256": "7813efbbed76cd1393af3dfc02479e88ab6f989f08a6b0124f1ad80549be9d13",
        "optimizer_executions": 39125,
        "train_batches": 313,
        "train_loss": 0.1734330690282048,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "ff86dcb42226ecf2700d8b6c556c310f106aaed2a7ca2c3a0cd900283ed3b4da",
          "prediction_sha256": "ffc454734deb12de64f8820d16acd6212bc993d3b51fb5965d4814e3dc431253",
          "samples": 1052,
          "val_mae": 0.10679974406957626
        }
      },
      {
        "epoch": 125,
        "global_step": 39438,
        "model_sha256": "2ee9096198d593f7af94f250432631c3e78d2d4a2ac3b15e9888ec423d6e307e",
        "optimizer_executions": 39438,
        "train_batches": 313,
        "train_loss": 0.1711849651733398,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "4ed68b824913d0e85c852f9a72461780067e666043136d4d21576c5df06b908e",
          "prediction_sha256": "dca2b67da5f43f8b133ac7fbef0ad6e7a09074bec45fa37b3564a66677703703",
          "samples": 1052,
          "val_mae": 0.1066889762878418
        }
      },
      {
        "epoch": 126,
        "global_step": 39751,
        "model_sha256": "3d9d747f5c71c8caebf94e931a99e73e003bc30dffa17cdb45435dc82092040d",
        "optimizer_executions": 39751,
        "train_batches": 313,
        "train_loss": 0.17100343398233403,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "a482a10ccba63f32d4e88b05790dca64168987962efe80b78e7005e71e986f2c",
          "prediction_sha256": "767592252ac28bef0edee55534dea029b10d4d86d07f9c2861dea75975532f12",
          "samples": 1052,
          "val_mae": 0.10753868520259857
        }
      },
      {
        "epoch": 127,
        "global_step": 40064,
        "model_sha256": "7677ddef84a05e88254280249096d1d136068b9be2f84c1e8f5745db28e28962",
        "optimizer_executions": 40064,
        "train_batches": 313,
        "train_loss": 0.17064362311622744,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "8c1188b86d228ddd8a85a39d5092d573e4f366dc3379447aeeb618955eb60fa9",
          "prediction_sha256": "239de2c30690d844685b4fe3a1d1a809b92be9efbd42e8598722dc26bf823eae",
          "samples": 1052,
          "val_mae": 0.1069079041481018
        }
      },
      {
        "epoch": 128,
        "global_step": 40377,
        "model_sha256": "8aa6fbf66bdbafd3ed78aff6cfb4c8261243d773ff82ceec197a3f0ba68afa7a",
        "optimizer_executions": 40377,
        "train_batches": 313,
        "train_loss": 0.16946798325663132,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "f682ed1697a4c4d264363fc79987f914d4e8b8f28218c3d6084d982e34d819d0",
          "prediction_sha256": "17f8883c6f5f26981b9564753370fcdba38fc3467c84b44ad8a66cf48f20fdcf",
          "samples": 1052,
          "val_mae": 0.10685945302248001
        }
      },
      {
        "epoch": 129,
        "global_step": 40690,
        "model_sha256": "988e69efef193c132973bcd0f3978ae4ab20de9f7cef11ee3048db3095c8e9c4",
        "optimizer_executions": 40690,
        "train_batches": 313,
        "train_loss": 0.1688614438987263,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "3ce05eaef1e1757e1845599e2c6e51eb06942d490f807fb656bb4e12ea5b83cf",
          "prediction_sha256": "b9848420b6b337d20fc6576cbac18e204b3f5488dcef17d1b66da968d4b01f70",
          "samples": 1052,
          "val_mae": 0.10663770139217377
        }
      },
      {
        "epoch": 130,
        "global_step": 41003,
        "model_sha256": "a31ea8aa6f26cb35cac528c2c82d8881fbfb68f0f7b0ac8cf6b5e6ef1ee8a907",
        "optimizer_executions": 41003,
        "train_batches": 313,
        "train_loss": 0.16759355086430713,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "e3226594e46a9fc3d1376ff8d2aa7bf2c0e07e649c9e13079d6d9d63e0da3c07",
          "prediction_sha256": "c80c7c604a1e15e9d53e00c7519e7ee79ffee3afdcf10b151d65f3ce436094ea",
          "samples": 1052,
          "val_mae": 0.10714296996593475
        }
      },
      {
        "epoch": 131,
        "global_step": 41316,
        "model_sha256": "83babc0ff2638a61ead26c2d4ea9a89cdb89abbbd96f99095203aa4f2250e353",
        "optimizer_executions": 41316,
        "train_batches": 313,
        "train_loss": 0.16702249127615923,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "91e50adbdbc8b074e310a2570257b4eb04712e9bc486307b5bad895b78b9d4a6",
          "prediction_sha256": "23def6163dd4d908ccf4fe2ac0befd543800a1ecf8914ca1f045d8c370862264",
          "samples": 1052,
          "val_mae": 0.10651274025440216
        }
      },
      {
        "epoch": 132,
        "global_step": 41629,
        "model_sha256": "e40a8dc53dbf0ff1b0b5b2907035f8b0930f9e05d6216bfd246bf61aba092ddc",
        "optimizer_executions": 41629,
        "train_batches": 313,
        "train_loss": 0.16648831385096416,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "03523d49f76bcbd790bf702caf9bf11d9fae1cf24c3acaf99cfd0672bd38daff",
          "prediction_sha256": "fe683e87ff99d9c85c4363b9b83ec95e928eb7c96d097d8d02aea37d851a523c",
          "samples": 1052,
          "val_mae": 0.10701462626457214
        }
      },
      {
        "epoch": 133,
        "global_step": 41942,
        "model_sha256": "83cc83a40e15e29c5de25997da9edbe2faeb0c64b9c04e97349bc2ac2d1379f1",
        "optimizer_executions": 41942,
        "train_batches": 313,
        "train_loss": 0.16553157098783347,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "14eaf308784ca4d0b00e1c11807dbcc5b698cdec9f37ca6e205c4063d903f94f",
          "prediction_sha256": "6ee8a695c4ff1af8c147bb46637c6db967866f1b0e2feb82546d817d3b52efe1",
          "samples": 1052,
          "val_mae": 0.10690569132566452
        }
      },
      {
        "epoch": 134,
        "global_step": 42255,
        "model_sha256": "5fa87e8fbd4f61756b541b9021440900bf98a7ec861cc43a91ec4398ed4851dd",
        "optimizer_executions": 42255,
        "train_batches": 313,
        "train_loss": 0.16527889269875665,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "00dbd2c3a1cdc69d532eef3ce36fb0103be9461ac103a7c8f125117346aabc3e",
          "prediction_sha256": "5a10d77232ca42a1aeeedb30995faeb6f95a125ada0632d1136baf95a2a98ff3",
          "samples": 1052,
          "val_mae": 0.10690917074680328
        }
      },
      {
        "epoch": 135,
        "global_step": 42568,
        "model_sha256": "1be3aa6731c08266e5b88c7a55d97530b9a7140a3dd356d17433b8300655986a",
        "optimizer_executions": 42568,
        "train_batches": 313,
        "train_loss": 0.16498154659394895,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "e1ab9677823aa36993801bc9f9055a9f4bd73bd366dc7229e331a9308e63597a",
          "prediction_sha256": "fcb31a0fbbdda76a26200136a28f6648c846c834704d4f60c2dbad8080c5ad2d",
          "samples": 1052,
          "val_mae": 0.10692676156759262
        }
      },
      {
        "epoch": 136,
        "global_step": 42881,
        "model_sha256": "bbc386833ca41199da825ddb1b53447f7e1c3f05c4a8797764d8b1796e3f7f38",
        "optimizer_executions": 42881,
        "train_batches": 313,
        "train_loss": 0.16416381404688782,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "880676619f57fbd831cb902ce51e89e09734cb6483c8713b1cb69c25c687137f",
          "prediction_sha256": "1c80f46289e12027b28a602e571a41361626ebaded80ca6c4b5b8a8c47dba9c8",
          "samples": 1052,
          "val_mae": 0.10659812390804291
        }
      },
      {
        "epoch": 137,
        "global_step": 43194,
        "model_sha256": "256bfa44616d3c2525b54af5e8136b5b0f893e0fb46cef6955a6cbb453dcc2d2",
        "optimizer_executions": 43194,
        "train_batches": 313,
        "train_loss": 0.16347791577255666,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "4b6fbaea72115fcd697a78ce3a088802285a57eeea71d5806d1fd79685673282",
          "prediction_sha256": "52dfc7122303eda3ac35f1a9f113a532bdceebfbe2ec95454e80b86e4c2aaab0",
          "samples": 1052,
          "val_mae": 0.10681605339050293
        }
      },
      {
        "epoch": 138,
        "global_step": 43507,
        "model_sha256": "4e077739a9958ea07f618d767ca503b08a06e8e032085823e1f1cfee613d5796",
        "optimizer_executions": 43507,
        "train_batches": 313,
        "train_loss": 0.16263123544784897,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "0968e41477b9986c459579f53898ffac42a61f5e77c9c17aed953228e6f248ed",
          "prediction_sha256": "ff7d157b0f0d71b16db8e2d963ec0e8c4ec93b39ccf8296150a6a0b5556957bb",
          "samples": 1052,
          "val_mae": 0.10659151524305344
        }
      },
      {
        "epoch": 139,
        "global_step": 43820,
        "model_sha256": "8619764a5b8826bde8ac3194b399ef791f2eb84bcc0fa49651ba69d82f85c476",
        "optimizer_executions": 43820,
        "train_batches": 313,
        "train_loss": 0.16266159909581002,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "27511d7f939f4536838236df3090fa9f8aa2e5ce4b2e322412f97fce5f368e0a",
          "prediction_sha256": "ff84cf0ad21926a4d886ebae36944eca768010dac60ec50bff3b5ef80c33bafb",
          "samples": 1052,
          "val_mae": 0.1065751239657402
        }
      },
      {
        "epoch": 140,
        "global_step": 44133,
        "model_sha256": "95fafd28b6bbca789b06c9a74ec1c503dcd865eb0848f397d8ca031dff61a3bb",
        "optimizer_executions": 44133,
        "train_batches": 313,
        "train_loss": 0.16174269395674518,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "bc562b3238bcfcfc9c15242f1b953c4c96d4ff58a66df9e4671eed8c423d256f",
          "prediction_sha256": "bf6bc7668a78deafdbe8684b632ef522f65ffffc60248dc3dc58e61ab8916dd5",
          "samples": 1052,
          "val_mae": 0.10648691654205322
        }
      },
      {
        "epoch": 141,
        "global_step": 44446,
        "model_sha256": "2acbe2bbd08c3324b9627794c6413d0bcacf9b5ef3ec81faac85e8e5a1698fc4",
        "optimizer_executions": 44446,
        "train_batches": 313,
        "train_loss": 0.1617456032540181,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "8f98e7e558f28d2441b9ff60b5fed522e02b22547e4587ee9cf68c3573291d08",
          "prediction_sha256": "1ef8b7bcd1d784c74710a829599dc93fc869bec951f163e5ade3e27278c31319",
          "samples": 1052,
          "val_mae": 0.10644268989562988
        }
      },
      {
        "epoch": 142,
        "global_step": 44759,
        "model_sha256": "d768bb8308c648487d70494d3e28a3d88785e0e09338f120214daf1aa44f068f",
        "optimizer_executions": 44759,
        "train_batches": 313,
        "train_loss": 0.16141036669979367,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "c98d4aef075a2e5c41523eb606e94e6f1d9ae9193f5ce1544e3617da24a5c829",
          "prediction_sha256": "83728968c080f0d1dad45009eb4f145e8511d717953b88c797593ac9db9b5fb0",
          "samples": 1052,
          "val_mae": 0.10676982998847961
        }
      },
      {
        "epoch": 143,
        "global_step": 45072,
        "model_sha256": "49e838f7822b14826784d3a272ec8db0f7465cbae27a9b21e0ed636c654b783c",
        "optimizer_executions": 45072,
        "train_batches": 313,
        "train_loss": 0.16059436210063965,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "3728abe2a8b016f9697b96cc817fd6502a5511222ae1fdb6b4012f848b9cabef",
          "prediction_sha256": "9036825ef6f31bdd5e616f5981f0c76014e2361b0db3e26ca9150e1b76e5fe11",
          "samples": 1052,
          "val_mae": 0.10722865164279938
        }
      },
      {
        "epoch": 144,
        "global_step": 45385,
        "model_sha256": "876039298e64fd7f8591ace60f33d2ff8b8ddf166b3fdc83aca8f12df0464414",
        "optimizer_executions": 45385,
        "train_batches": 313,
        "train_loss": 0.16061636737075372,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "f3c2f42b11a0347e056d1b4ab930cf192d3ec9162a5e7369bb051f306ece3eb8",
          "prediction_sha256": "944b903da60011738d43475680129c21cb7f0e0d4b449c2dcfd10b44dd983767",
          "samples": 1052,
          "val_mae": 0.10630998760461807
        }
      },
      {
        "epoch": 145,
        "global_step": 45698,
        "model_sha256": "6c8033f7ef0a8c6d846a97249f09cfaf76f1239d18be3c10284b2ab1d61fac48",
        "optimizer_executions": 45698,
        "train_batches": 313,
        "train_loss": 0.16039679132787174,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "b35f9e005051cac6ed6380344c2d4b54fea08522566b842b7b27c06c28cbbbe8",
          "prediction_sha256": "366161c748d1d26ccec23ab6891ef5b2adb57838a59512343416c5062e81d1c5",
          "samples": 1052,
          "val_mae": 0.10688202828168869
        }
      },
      {
        "epoch": 146,
        "global_step": 46011,
        "model_sha256": "f128e48b96a261836bc1da5822aa3c2b6015ba639755bbb41235068208671f4b",
        "optimizer_executions": 46011,
        "train_batches": 313,
        "train_loss": 0.16037579859965168,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "8fccc263bb1c3107a830a178215243bec08a79114300e4ed7a85be56e51b4354",
          "prediction_sha256": "174249b2c9f4641ab85ba05eea967010dbfb625b1f0a93745a95bf23d3a6bbc4",
          "samples": 1052,
          "val_mae": 0.1066567450761795
        }
      },
      {
        "epoch": 147,
        "global_step": 46324,
        "model_sha256": "a28e555cce7ed0d778045adc6beb1634269d25bce44860a1926b724f44bf37f2",
        "optimizer_executions": 46324,
        "train_batches": 313,
        "train_loss": 0.15957019457211014,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "386b6a51b8c7785ce5c9f4ae3fa6432b642c7c6a5c2ca2dd4f74a1d79728e432",
          "prediction_sha256": "b4e0537a4d27a5945da751457ea9165605d8bac385c693d746e2f32ad71b8911",
          "samples": 1052,
          "val_mae": 0.10661549121141434
        }
      },
      {
        "epoch": 148,
        "global_step": 46637,
        "model_sha256": "0a6816e976a27ba624fa2129b9e3e0282b8e8aee9de30c5866250b2fcace0fd0",
        "optimizer_executions": 46637,
        "train_batches": 313,
        "train_loss": 0.15935614684094843,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "f46d057bb8de0a7ed48d5df9a95b42917d41c27d376888bb3043a3ce1461f994",
          "prediction_sha256": "48a9f5e4d57924a26dd52885284b296307690f55a3d0e4c3eee36e942fd842b9",
          "samples": 1052,
          "val_mae": 0.10634325444698334
        }
      },
      {
        "epoch": 149,
        "global_step": 46950,
        "model_sha256": "64b6fd976f088200dd0661b253798de4b7e536982bceda57c3d450cc237599a6",
        "optimizer_executions": 46950,
        "train_batches": 313,
        "train_loss": 0.15976242529061654,
        "train_samples": 62470,
        "validation": {
          "pose_sha256": "79269e1e60ce6cf06caf38f96178f3b8332fb086c2b37b824bd4c0dbb6877e19",
          "prediction_sha256": "90b5d8aced8980a07fca29d9ba58860a4bfa0e7b522d2e14dc2841f1bb865f70",
          "samples": 1052,
          "val_mae": 0.1064886674284935
        }
      }
    ],
    "fit_result_sha256": "bb4aa8f8c67cb45e9c5917a33282364e869bbfef1907e41105804fb35cb3d15e",
    "format": "original_rn18_fusion_native_curve_data_v1",
    "journal_sha256": "20f0d193106e13645dc6a11cef0cb2a11c5e218f1bf047a1c65fd59d380c2232",
    "selected_best": {
      "checkpoint_sha256": "b37f0f3c36ce48ecdcba43428db5529195bc08abfe7c7f5c06d5d4842b91dcd2",
      "file": "rn18-s-8ch-centerfusion-epoch=144-val_mae=0.1063.ckpt",
      "score": 0.10630998760461807,
      "validated": true
    },
    "selected_epoch_candidates_zero_based": [
      144
    ],
    "selected_epoch_zero_based": 144,
    "training_unit": "native sample-weighted composite loss",
    "validation_unit": "radian; original sample-weighted val_mae"
  },
  "fixed_agreement_policy": {
    "native_aggregation": "sample-count weighted original per-batch masked L1, float32 accumulation",
    "native_mae_atol_rad": 2e-06,
    "native_mae_rtol": 2e-06,
    "pooled_aggregation": "float64 absolute-error sum / exact valid-angle count",
    "prediction": "exact typed prediction/target/mask/input replay",
    "saved_fit_prediction": "exact typed selected-fit validation pose agreement before test",
    "scope": "fixed before validation; no test-driven tolerance or checkpoint changes"
  },
  "format": "original_rn18_fusion_center_completed_report_v1",
  "full_paper_complete": false,
  "independent_test_sha256": "c0a8b2c81f510f98586dd95a195abe6ab80f51987440753f403682e70bbf39d6",
  "initializers": {
    "emg": {
      "checkpoint_selection": "native validation BEST",
      "completed_epochs": 150,
      "export_file_sha256": "4d53b5c6c548a16e29f466cd1917c2be62246adeb9cf0f730febcde3f71a5922",
      "export_lineage_sha256": "ea6cedbb668165b5bf7cad45870284bd645931fe4ced52f17e637e39e4705e40",
      "format": "rn18_fusion_exact_cotrain_best_admission_v1",
      "full_initializer": true,
      "full_paper_complete": false,
      "fusion_cohort_sha256": "1736318c9faf5740417ca676cd520d47ada84d16eac31f865e9cdf24a9f2e38c",
      "future_fusion_partition_content_sha256": "44e9ebc8a2f63a79699c5a976bf2717b0fe7a376427c7c46f538d0745c2b4bd0",
      "input_binding": {
        "cohort_sha256": "7cb310e2d5bf564a5317db018466def1c6a725057251b181daf1e99eee992a55",
        "footprints": {
          "holdout_union_rows": {
            "test": 16752000,
            "val": 6312000
          },
          "overlap_rows": {
            "test": 0,
            "val": 0
          },
          "train_union_rows": 41609029
        },
        "partition_content_sha256": "ca21b51f8206812c0d886238d2bb47657678b7514d53b413dbb4161bbfbbddfc",
        "producer_only_payload_files": [
          "field:label_gesture_active",
          "field:label_gesture_class",
          "field:mocap_left_wrist_angles_valid",
          "field:mocap_right_wrist_angles_valid"
        ],
        "producer_only_payload_sha256": {
          "field:label_gesture_active": "27bfef6dc99d98110613acc8685dad1ee5321f119098d9bac772c27d4f77d7cb",
          "field:label_gesture_class": "55a07d9d639483e54eef963a527d50554902d632eea4e42a29651f40f83b4212",
          "field:mocap_left_wrist_angles_valid": "d48e0430adac0b7ffd0e048e05cdd0fc084c90f28387ca79406dbf62bc9d9389",
          "field:mocap_right_wrist_angles_valid": "6175793773e197b2cdb825d0b131ad686afeaeb1180b9852e0953b7fd986349d"
        },
        "shared_payload_files": 12,
        "shared_payload_sha256": "49fd620112b6bc5cbc4644407fb7d05b64443744e42e5e10ae13cd2132eeec07"
      },
      "model_state_sha256": "7c5fa2c7faa93bf75c1545a3bcce4b9a0bc43876ba477f3b111086c34b1ba223",
      "optimizer_executions": 36150,
      "producer_family": "rn18-fixed-holdout-cotrain-small-150-0.01",
      "producer_reader_sources_sha256": {
        "egoemg_repro/classic_emg_train.py": "52b1ce655f3c4e107e3d809e403cd507fbcfd655f7ac9ea7bc0b9a592b18b0d8",
        "egoemg_repro/emg_initializer_runtime.py": "4286435f60ca50499d88e29d38ed66e3b1bf965039024ddb5dadb736d3fa5ff9",
        "egoemg_repro/rn18_initializer_cohort.py": "3a57e264fc316578c44aa813a6ae153854911373bb3b61edcf0750a8b8544be8",
        "egoemg_repro/rn18_initializer_data.py": "700afbc082841236792900d9856bbc26bca7c870dc47ccf4c22f78a2a58bccff",
        "egoemg_repro/rn18_initializer_train.py": "3f3270d2fe41eab05048ab2d271582e814cf18551cb432fa258947a7fefb501e",
        "egoemg_repro/vision_literal_check.py": "9601034285aaea5cb35abed793576ab5ecf901146b55a4c3901f7b242234b227",
        "egoemg_repro/vision_literal_runtime.py": "bb430267f819c1a1c46ed961397a81c3b9619d9a48f321939cc78808bc85e4d0",
        "egoemg_repro/vision_literal_train.py": "b89e20be1e6ef3669f634391ac072c51ecc929c9517fbc244b6a02747c52efd7"
      },
      "scientific_eligible": false,
      "scope": "full",
      "selected_epoch": 74,
      "selected_score": 0.24354727566242218,
      "sha256": "f05306e3e66249768cf883d78acf75b3971b17ae43781b8514d55a463ea516db",
      "source_cohort_sha256": "7cb310e2d5bf564a5317db018466def1c6a725057251b181daf1e99eee992a55",
      "source_fit_identity_sha256": "b4f12902824d1a23dca3e9214da29d3581a75f1a8320543cced8c84f43f8c735",
      "source_fit_result_file_sha256": "908a9422e6236a3f9041a8a1487719370d8a0ec349d6c1301c1cd021051d4c81",
      "source_selected_checkpoint_sha256": "914af66d516bcba3dc6e09742fc1135af64e4e04e874ece4d63abf59030f532d",
      "sources_sha256": {
        "egoemg_repro/fusion_initializer_handoff.py": "9fdf4842d12b61ed5092746b06a0284af4a2abca91c93a388c67fcda930b5f80",
        "egoemg_repro/fusion_literal_runtime.py": "3b357bdb0e41f3536c2296324917465ad25b2d13c8cccb553815fb729310ca60",
        "egoemg_repro/rn18_fusion_cohort.py": "0eca1ac9b5e46421b667d4a50539b14d4a9df6f70cce04d383d7b0f5a4ae225a",
        "egoemg_repro/rn18_fusion_data.py": "c9f90ced4afb7868f99510632a9daacb4b86fd7a81265d612468b77faa1c205d",
        "egoemg_repro/rn18_fusion_literal.py": "1160b90b66b8a9155c4ddad17e7f7cba1322d3f146b6da7011dc5539214fd5e7",
        "egoemg_repro/rn18_fusion_runtime.py": "c0b4d3ffaddef4b1b9c70146c7d21c085bdda257e2326074749bc8189de076ef",
        "egoemg_repro/rn18_fusion_trained_handoff.py": "54d957d58d1b28ebe827774a2df406545504a0e971b09099990f77032354bb41",
        "egoemg_repro/rn18_fusion_transfer.py": "01831c0bdc3974a90b215cf84d1e4f43d05082f3d0f71a4a896b6f90e811cf34",
        "egoemg_repro/rn18_fusion_vision_producer.py": "46080ff9dc1f491ecf1375109ec191204c38f032f2929cb4de62777f1fd10801",
        "egoemg_repro/rn18_initializer_cohort.py": "3a57e264fc316578c44aa813a6ae153854911373bb3b61edcf0750a8b8544be8",
        "egoemg_repro/rn18_initializer_data.py": "700afbc082841236792900d9856bbc26bca7c870dc47ccf4c22f78a2a58bccff",
        "egoemg_repro/rn18_initializer_train.py": "3f3270d2fe41eab05048ab2d271582e814cf18551cb432fa258947a7fefb501e"
      },
      "tensor_inventory": {
        "model.decoder.encoder.layers.0.linear1.bias": {
          "dtype": "torch.float32",
          "sha256": "8a6862cd1126baa40c84dd28335e706e473fa277dc559127aacd8cc9de5ffa16",
          "shape": [
            512
          ]
        },
        "model.decoder.encoder.layers.0.linear1.weight": {
          "dtype": "torch.float32",
          "sha256": "265a866302b31bd6217f721c90f07700fd5be2747858dc068c3e4212bd653117",
          "shape": [
            512,
            256
          ]
        },
        "model.decoder.encoder.layers.0.linear2.bias": {
          "dtype": "torch.float32",
          "sha256": "2e3c7977845817990aaf8f0e0e7c554713e2ac685bbf7c0990ce4de216680990",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.0.linear2.weight": {
          "dtype": "torch.float32",
          "sha256": "a3b66d683005ebbc8f35a1f93366ed0ce8de740fccdf17544021f1df2dc00137",
          "shape": [
            256,
            512
          ]
        },
        "model.decoder.encoder.layers.0.norm1.bias": {
          "dtype": "torch.float32",
          "sha256": "8465e2df409a3d848bf8bad3085b9649f520039975d5192d91aeb9e91dea7601",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.0.norm1.weight": {
          "dtype": "torch.float32",
          "sha256": "c52239221eb30d9086238f611737b71d03b05cd812c8e502e13d45c6f1cec351",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.0.norm2.bias": {
          "dtype": "torch.float32",
          "sha256": "795ba038210fdbda420bf35624bfea962a707510db4958050918a316d8011d3c",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.0.norm2.weight": {
          "dtype": "torch.float32",
          "sha256": "964dbe639d5f7ae8846408c73ebdf7684c6168c3ddd86d9008ec2c4b3d5f86b5",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.0.self_attn.in_proj.bias": {
          "dtype": "torch.float32",
          "sha256": "45b7ef4f6a4b96ad77472c6a78163cb8f95e99197720a79032ace880effcbf50",
          "shape": [
            768
          ]
        },
        "model.decoder.encoder.layers.0.self_attn.in_proj.weight": {
          "dtype": "torch.float32",
          "sha256": "1c78f33d017bf3ac4b2c2d1f0cd15f9e0d05e4bb3c9e7d9087d647a14b876f52",
          "shape": [
            768,
            256
          ]
        },
        "model.decoder.encoder.layers.0.self_attn.out_proj.bias": {
          "dtype": "torch.float32",
          "sha256": "ae20fe45a71003358b383adfaa5bfe1b7eb091eab5cd2b7382a593842495a969",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.0.self_attn.out_proj.weight": {
          "dtype": "torch.float32",
          "sha256": "1caa768cd22e67e57cea16fb8203f54dbcc4e46c567711e6279f4060f80b8f62",
          "shape": [
            256,
            256
          ]
        },
        "model.decoder.encoder.layers.1.linear1.bias": {
          "dtype": "torch.float32",
          "sha256": "02e1141daaaa1b70ed8b9db8cadcfffc886c26a5df9c5b4ef418f3f3fca27abd",
          "shape": [
            512
          ]
        },
        "model.decoder.encoder.layers.1.linear1.weight": {
          "dtype": "torch.float32",
          "sha256": "2df5818925c3c155cb9e3c0cf792831c3e5f590e93e0486a9eefa6d1935d68bf",
          "shape": [
            512,
            256
          ]
        },
        "model.decoder.encoder.layers.1.linear2.bias": {
          "dtype": "torch.float32",
          "sha256": "ff9356cdc1bdad4bc874c667611700ca7f8b5a40f8a90272534cdeb2551264e7",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.1.linear2.weight": {
          "dtype": "torch.float32",
          "sha256": "25e6fad6319a53948068bdd3b31c6522e174c1e341f1b0dad3ec176e7ca50fb5",
          "shape": [
            256,
            512
          ]
        },
        "model.decoder.encoder.layers.1.norm1.bias": {
          "dtype": "torch.float32",
          "sha256": "42527c6699aa32bf8ccd4cc8f632b819570d2414d00d4c9956514b9c92762a7f",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.1.norm1.weight": {
          "dtype": "torch.float32",
          "sha256": "970acfef9d84497ca636c55fa036b6429cb657830fe85e9b3efde37b8b7c9dfa",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.1.norm2.bias": {
          "dtype": "torch.float32",
          "sha256": "a04dc7ce118bd68c5b14112f96b696b6b222cae0705dbe1522f5e29759ad276c",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.1.norm2.weight": {
          "dtype": "torch.float32",
          "sha256": "30b1ac57ebb15601172d62332d58c8728d94a34e5044ec61e3180954c38c7543",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.1.self_attn.in_proj.bias": {
          "dtype": "torch.float32",
          "sha256": "62e51ad431ecabfb14ac89b08765affbc09be12f6885e50b7e62a7baff39674c",
          "shape": [
            768
          ]
        },
        "model.decoder.encoder.layers.1.self_attn.in_proj.weight": {
          "dtype": "torch.float32",
          "sha256": "172043c16522108edb50b9c9c3c7e6bab94a52790e1e147540efa45560bc2c85",
          "shape": [
            768,
            256
          ]
        },
        "model.decoder.encoder.layers.1.self_attn.out_proj.bias": {
          "dtype": "torch.float32",
          "sha256": "7d762cf3ffd058b90bc6e7896b31c0e3c8f9bf92f1ef573a821f13af72491591",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.1.self_attn.out_proj.weight": {
          "dtype": "torch.float32",
          "sha256": "7cc2be267ca2727afd859a4ac448c827494ac3c111716b247c40ed5616737c33",
          "shape": [
            256,
            256
          ]
        },
        "model.decoder.encoder.layers.2.linear1.bias": {
          "dtype": "torch.float32",
          "sha256": "0ce2ae609ae640569afeed07937adf0a10d85926f1688707988f9d1a96cf8063",
          "shape": [
            512
          ]
        },
        "model.decoder.encoder.layers.2.linear1.weight": {
          "dtype": "torch.float32",
          "sha256": "2353452d1284202a4ae7de06396d259f586b0d58916d76d4fea900cfac32af04",
          "shape": [
            512,
            256
          ]
        },
        "model.decoder.encoder.layers.2.linear2.bias": {
          "dtype": "torch.float32",
          "sha256": "1f8195ca618128c054b32c8fc56619e53dc8eac68f34aac4cfd650d2f2757b89",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.2.linear2.weight": {
          "dtype": "torch.float32",
          "sha256": "d934b56de7958740d463dbb0d5ec1ec41e314195c78d61c909d6d8b54af837f0",
          "shape": [
            256,
            512
          ]
        },
        "model.decoder.encoder.layers.2.norm1.bias": {
          "dtype": "torch.float32",
          "sha256": "8d3afc4ab250c7983e7853b9f86671f2a0bb39a784aa96266220893925b556a5",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.2.norm1.weight": {
          "dtype": "torch.float32",
          "sha256": "f8a6659762a807f3b5f03b3cb1f7112ce0666346f8c7c8753ca83f0da01bdf32",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.2.norm2.bias": {
          "dtype": "torch.float32",
          "sha256": "b1990518b839ea31a3ae4fbd13440b35e4c90c24616d74c3a31ee9bc9b37f065",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.2.norm2.weight": {
          "dtype": "torch.float32",
          "sha256": "579b4009909ab8357f9cda280cfb8d10df73d38d9b850e815891d7b8c7f0e08e",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.2.self_attn.in_proj.bias": {
          "dtype": "torch.float32",
          "sha256": "39643d15afd13acb5f5c6748d925078d6657f4b310986b7e63c91e514c5c29ff",
          "shape": [
            768
          ]
        },
        "model.decoder.encoder.layers.2.self_attn.in_proj.weight": {
          "dtype": "torch.float32",
          "sha256": "546650d1fe1b091ef0d1c8016c2b2df265c74e1e83466b7d9df496c7fc3b733e",
          "shape": [
            768,
            256
          ]
        },
        "model.decoder.encoder.layers.2.self_attn.out_proj.bias": {
          "dtype": "torch.float32",
          "sha256": "c71474b166b3c1f2a6ff2bc92f671a1d1505f776cf218794b524e12af280e710",
          "shape": [
            256
          ]
        },
        "model.decoder.encoder.layers.2.self_attn.out_proj.weight": {
          "dtype": "torch.float32",
          "sha256": "ed2f1130bc1e172de941052f80a389ab3b774ac290ffeaec116890865965d30c",
          "shape": [
            256,
            256
          ]
        },
        "model.decoder.output_proj.bias": {
          "dtype": "torch.float32",
          "sha256": "0293393c8d857add30121da1aac6d041481be1e8ae68c8700e274fc2c94d1320",
          "shape": [
            256
          ]
        },
        "model.decoder.output_proj.weight": {
          "dtype": "torch.float32",
          "sha256": "f96201e2b782b95b60bc70a588c36b92f313f2f4bc38a2779425a717b33098aa",
          "shape": [
            256,
            256
          ]
        },
        "model.featurizer.layers.0.conv.0.bias": {
          "dtype": "torch.float32",
          "sha256": "12434a5cceb626f729afa4ad9f6be28894dbec224b9d4ba10ad4f7d227e14e1d",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.0.conv.0.weight": {
          "dtype": "torch.float32",
          "sha256": "6f1556026df16f322bd6f25a9ff7b8e0e59ecf01a0b91eb189c6614a122ad244",
          "shape": [
            256,
            8,
            11
          ]
        },
        "model.featurizer.layers.0.norm.bias": {
          "dtype": "torch.float32",
          "sha256": "cc2fd73eeb317d2c0288c6cafe721c7ab2f4f6e979d6f1f30bea0828e6043f19",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.0.norm.weight": {
          "dtype": "torch.float32",
          "sha256": "e0e49f5ee7ce511e147db002b568fe0c35bf1c10d697bf1105502fe62dfca3d6",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.0.se.net.0.bias": {
          "dtype": "torch.float32",
          "sha256": "9b4a8415b6719867ddde42a40fa04f46ba10b62c205cd1d15bb0a48ab3ca600d",
          "shape": [
            64
          ]
        },
        "model.featurizer.layers.0.se.net.0.weight": {
          "dtype": "torch.float32",
          "sha256": "1f8442488ad6658b2e6ae4ef2829d284fb67c9e8ec6b57f860edc08ac4825018",
          "shape": [
            64,
            256,
            1
          ]
        },
        "model.featurizer.layers.0.se.net.2.bias": {
          "dtype": "torch.float32",
          "sha256": "e48e43b26bb25ae44ff3984dace44c380e3489c6cb1add2bd40431d0e910edcc",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.0.se.net.2.weight": {
          "dtype": "torch.float32",
          "sha256": "9cd50132b58b3c69bc35ac974e8bcde8b17d3169a7f15d34572494cb2d64db89",
          "shape": [
            256,
            64,
            1
          ]
        },
        "model.featurizer.layers.1.conv.0.bias": {
          "dtype": "torch.float32",
          "sha256": "747c56e9d7193341a7fbe3205814e90dd1bbdc24ac97031857ff77fbb677f7c1",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.1.conv.0.weight": {
          "dtype": "torch.float32",
          "sha256": "d01fded5452a7674a9640f682d7dde22ef4727ed2536deb12ff2be093ec80885",
          "shape": [
            256,
            256,
            5
          ]
        },
        "model.featurizer.layers.1.norm.bias": {
          "dtype": "torch.float32",
          "sha256": "6635cb1efff75b1a32d00d0627259e5b66ff829fd353fec713a9966b2492b79f",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.1.norm.weight": {
          "dtype": "torch.float32",
          "sha256": "80b5a7a346cfbb894c164d6355b6bd3e7ab90f0fe4850123fd358858d40ef0c0",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.1.se.net.0.bias": {
          "dtype": "torch.float32",
          "sha256": "da0fce3c3566dfec4469f916d8bfee4ee02ac5a6f485fe9bcf50d326eef8edb5",
          "shape": [
            64
          ]
        },
        "model.featurizer.layers.1.se.net.0.weight": {
          "dtype": "torch.float32",
          "sha256": "cb3eb0ed1a6f73005f4e6a801331fedf6c95c743885b487e66238b854b2eb219",
          "shape": [
            64,
            256,
            1
          ]
        },
        "model.featurizer.layers.1.se.net.2.bias": {
          "dtype": "torch.float32",
          "sha256": "cc0595010be85d9b5ca917c430db5f7b4e313a8def13a054e36a8b88dce0791a",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.1.se.net.2.weight": {
          "dtype": "torch.float32",
          "sha256": "e11e985cd820317e8c0dc73772769606f85b7eef45a6722040b665138d16e5d0",
          "shape": [
            256,
            64,
            1
          ]
        },
        "model.featurizer.layers.2.layers.conv1dblock.conv.0.bias": {
          "dtype": "torch.float32",
          "sha256": "bfedae3e43ef6de7eba7459d9b99583348618947bbd32aa496b85bf48d60296b",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.2.layers.conv1dblock.conv.0.weight": {
          "dtype": "torch.float32",
          "sha256": "b1de1377f161100c28c1c39ccc7a1bf447d8a8cd941f837fcaed2654553574e5",
          "shape": [
            256,
            256,
            9
          ]
        },
        "model.featurizer.layers.2.layers.conv1dblock.norm.bias": {
          "dtype": "torch.float32",
          "sha256": "a1eaf494b473fda7139f90e52a5f78a5ba6783eaca689591e720b891081520c4",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.2.layers.conv1dblock.norm.weight": {
          "dtype": "torch.float32",
          "sha256": "a35daac49fae663c68eaccbfe34b1ff5a8c7374c2bf119919a5d3b5d6dc003fb",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.2.layers.conv1dblock.se.net.0.bias": {
          "dtype": "torch.float32",
          "sha256": "9a235e98c8ae923e25e2e4a89a6fdbaf83f7d1e92e2f12f6c8cf266b3104cdca",
          "shape": [
            64
          ]
        },
        "model.featurizer.layers.2.layers.conv1dblock.se.net.0.weight": {
          "dtype": "torch.float32",
          "sha256": "80621d908322b64e50057e4c0556af23049025b8a173690371094ff51ece0219",
          "shape": [
            64,
            256,
            1
          ]
        },
        "model.featurizer.layers.2.layers.conv1dblock.se.net.2.bias": {
          "dtype": "torch.float32",
          "sha256": "b0d0439d884f9b9c9066f8e4be018879e5bb77115250477de2d7a85c5379475b",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.2.layers.conv1dblock.se.net.2.weight": {
          "dtype": "torch.float32",
          "sha256": "0b284a7e076f620c44b8b043df99ae500c62f7d7986c8cb2f5289928ea6fc17d",
          "shape": [
            256,
            64,
            1
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.0.conv2d.bias": {
          "dtype": "torch.float32",
          "sha256": "1dd221a14c5f7ad508207ebd5874f42c04821b1b258bcb5bfd03122dc1c743b7",
          "shape": [
            8
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.0.conv2d.weight": {
          "dtype": "torch.float32",
          "sha256": "fa50d2142ed49b79f0ae53bbf2284a0cd939538b7a69c3666ff74a869d9ecb16",
          "shape": [
            8,
            8,
            1,
            5
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.0.layer_norm.bias": {
          "dtype": "torch.float32",
          "sha256": "e811f694c33d408d830fc665901025265a13d46f89fd0eab84ae3559895a8cb1",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.0.layer_norm.weight": {
          "dtype": "torch.float32",
          "sha256": "22365908f62432bb37762ec58187f5ac8c1ac8e481f4bbe731a3f711235f3aec",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.0.se.net.0.bias": {
          "dtype": "torch.float32",
          "sha256": "63bdc4021f425c6f19c9096ffdd6cd453c62d7ee2cde14e60992d39e139f61db",
          "shape": [
            64
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.0.se.net.0.weight": {
          "dtype": "torch.float32",
          "sha256": "811f3158d48c898a8aa23777f83585d48a619a85d36c2bb34b88150e18fbe889",
          "shape": [
            64,
            256,
            1
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.0.se.net.2.bias": {
          "dtype": "torch.float32",
          "sha256": "24f9dfa938be5b88cb0f6e42bbdf580b911c0169757b4a15376933dfd1b2e3c7",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.0.se.net.2.weight": {
          "dtype": "torch.float32",
          "sha256": "63527755bd8e9cc85764d6737a9df054bb65190bbc42bc28d103b295426c36bf",
          "shape": [
            256,
            64,
            1
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.1.fc_block.0.bias": {
          "dtype": "torch.float32",
          "sha256": "869ff7747910b1bda2c88c0d8ee9aefa1c8b085bf076c43fa30deb15fbea61a4",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.1.fc_block.0.weight": {
          "dtype": "torch.float32",
          "sha256": "1e116867b5596a47f4a941228f210e277adbe8dbb309b62c8ba9b5e99508188b",
          "shape": [
            256,
            256
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.1.fc_block.2.bias": {
          "dtype": "torch.float32",
          "sha256": "da29bfb3bee3c8ca85aea3a62945b74c77d5ffd9b8d0874a7d9829b153e5c7c0",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.1.fc_block.2.weight": {
          "dtype": "torch.float32",
          "sha256": "bcf812528bc7a985847dd69e6cd2fc17e7e33d7b5363f75567b2b09072e2b1bc",
          "shape": [
            256,
            256
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.1.layer_norm.bias": {
          "dtype": "torch.float32",
          "sha256": "f6a8bfb30886625dc74105bef5153598a14fe62871cf7f6af638c84cfccb4cd3",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.1.layer_norm.weight": {
          "dtype": "torch.float32",
          "sha256": "3e271981f37eb5c1f637a7673d691ef71d685628395fa3dc8c54acf2b908cd56",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.1.se.net.0.bias": {
          "dtype": "torch.float32",
          "sha256": "53b200216daae86ac30b586bfe16b00e3e1306031187824d7b74d4f722b7c491",
          "shape": [
            64
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.1.se.net.0.weight": {
          "dtype": "torch.float32",
          "sha256": "c297f820bf668b70148064248cf8864250c9edb6ee62e09115e91988b04b3b73",
          "shape": [
            64,
            256,
            1
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.1.se.net.2.bias": {
          "dtype": "torch.float32",
          "sha256": "ccf94da7065281268f1fe09aabd9237898676f527a0a2aa6445c12f5fc4976e7",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.2.layers.tds_block.tds_conv_blocks.1.se.net.2.weight": {
          "dtype": "torch.float32",
          "sha256": "a8ec16aa098e2e2f55aced78da9fdc73ec140a9e26a3eab9dc16c31694d3749b",
          "shape": [
            256,
            64,
            1
          ]
        },
        "model.featurizer.layers.3.layers.conv1dblock.conv.0.bias": {
          "dtype": "torch.float32",
          "sha256": "b8cbc9bd24d77aa93312b1a3b763ba49689b8b6c4ded7a510d7d2c94a3a262b3",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.3.layers.conv1dblock.conv.0.weight": {
          "dtype": "torch.float32",
          "sha256": "b033f0c87d87eaf9dd6c69793e6dda1b80b6ea3bdeb707051408f97da6a05585",
          "shape": [
            256,
            256,
            3
          ]
        },
        "model.featurizer.layers.3.layers.conv1dblock.norm.bias": {
          "dtype": "torch.float32",
          "sha256": "16a00aaffb02b21632d7b80ac446e767d7f23a5056733bf6b60e1481a10d53be",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.3.layers.conv1dblock.norm.weight": {
          "dtype": "torch.float32",
          "sha256": "823723cd28a51a90e3cefa0e7a2d6e592989417a5d59f6fe69f735d869eda953",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.3.layers.conv1dblock.se.net.0.bias": {
          "dtype": "torch.float32",
          "sha256": "ab9628821b7b27972d8d2ce672bab04819c249c247220d301abb9eeae0efdffa",
          "shape": [
            64
          ]
        },
        "model.featurizer.layers.3.layers.conv1dblock.se.net.0.weight": {
          "dtype": "torch.float32",
          "sha256": "f145dc2a355ec1a8e44c9b71c09cea8600950dbed566602d375bc0646ee17399",
          "shape": [
            64,
            256,
            1
          ]
        },
        "model.featurizer.layers.3.layers.conv1dblock.se.net.2.bias": {
          "dtype": "torch.float32",
          "sha256": "23940b76281106c5cb10de3c37c81e241ce628f7a97db8d4cddde8e15b024508",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.3.layers.conv1dblock.se.net.2.weight": {
          "dtype": "torch.float32",
          "sha256": "45e8154c1075854922dc8dd4aeb85d0041274e29f485d54a6d996c2677be0fbe",
          "shape": [
            256,
            64,
            1
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.0.conv2d.bias": {
          "dtype": "torch.float32",
          "sha256": "b2c38bfc2c7f136eb97b389592c09fe198df9378f83bb6cdf38508d2d26bf899",
          "shape": [
            8
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.0.conv2d.weight": {
          "dtype": "torch.float32",
          "sha256": "53a878b565e1c4dfb086b8cd87631221a8d3731c63788022a6518ffa76bc98f9",
          "shape": [
            8,
            8,
            1,
            3
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.0.layer_norm.bias": {
          "dtype": "torch.float32",
          "sha256": "9c694513db185e4dac74c664a4f425fbd9b579ac76dc6239d7d545d5db05b195",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.0.layer_norm.weight": {
          "dtype": "torch.float32",
          "sha256": "46ed276c55954c790bc4a29da62b99cb7a825983105b250bd0d6199817a83623",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.0.se.net.0.bias": {
          "dtype": "torch.float32",
          "sha256": "6a4c3e276af176a15f3e8cdecbe09a3e01e4e075e838c221911e5bb1caddbaab",
          "shape": [
            64
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.0.se.net.0.weight": {
          "dtype": "torch.float32",
          "sha256": "41692839e8e7d91d5f45719950dbf01ff00c2b1527a71e6cc47484f5a56d6e74",
          "shape": [
            64,
            256,
            1
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.0.se.net.2.bias": {
          "dtype": "torch.float32",
          "sha256": "747f8ef49f2fe0aff66cadf1e80972e55be7134848a6b18b9fcceb16482b3c3d",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.0.se.net.2.weight": {
          "dtype": "torch.float32",
          "sha256": "43385c670f5fc07fe30a6d37e35ddcbf88f609da2d1c16e2694927acce0c0ee8",
          "shape": [
            256,
            64,
            1
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.1.fc_block.0.bias": {
          "dtype": "torch.float32",
          "sha256": "4f61537066d62e333ac1cbca7fb1b318c386b4093bbe49e7fd30cdc0dc302053",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.1.fc_block.0.weight": {
          "dtype": "torch.float32",
          "sha256": "6f299984f0d2e5850c0525d80d35b3d5a903df39380fe334a2aa5726ba8f3cc9",
          "shape": [
            256,
            256
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.1.fc_block.2.bias": {
          "dtype": "torch.float32",
          "sha256": "d2201a2e9f064b3d0de8b0f0deba457ec373c6b1713fc6217661928b308e8384",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.1.fc_block.2.weight": {
          "dtype": "torch.float32",
          "sha256": "824ca5403c710a224756cc6ed6816c88636ca69272e54f2089f749e5b0b09b9b",
          "shape": [
            256,
            256
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.1.layer_norm.bias": {
          "dtype": "torch.float32",
          "sha256": "5b9710174acb1885bad82090c2d85cb2d968d3c5462908f3cb941bb11d1d5d0f",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.1.layer_norm.weight": {
          "dtype": "torch.float32",
          "sha256": "a4e2aeee43ecbb3dc64ea44cc987eca82e31a06864da8fbaa1872ac964bb9a55",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.1.se.net.0.bias": {
          "dtype": "torch.float32",
          "sha256": "213b34f7fe807a77ccac263ebe58b5336be37747b89f375e812a278216d51e0e",
          "shape": [
            64
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.1.se.net.0.weight": {
          "dtype": "torch.float32",
          "sha256": "b00f167775d3064867654fa6ce8993c0d2d5ae3747b7c7baf00317a312f5d590",
          "shape": [
            64,
            256,
            1
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.1.se.net.2.bias": {
          "dtype": "torch.float32",
          "sha256": "582e1cd5458975864322acd5b45024ef1a3a136330ea759c814f4ef7a0cd8c49",
          "shape": [
            256
          ]
        },
        "model.featurizer.layers.3.layers.tds_block.tds_conv_blocks.1.se.net.2.weight": {
          "dtype": "torch.float32",
          "sha256": "7d369e325a6b78bd7056c7010b1264903535d8a872c8210b64f4d03470c9801e",
          "shape": [
            256,
            64,
            1
          ]
        },
        "model.head.net.0.bias": {
          "dtype": "torch.float32",
          "sha256": "e84e7662ece2d37490849944be03f1d02e8c0054a880093a8498f80be523bd9c",
          "shape": [
            512
          ]
        },
        "model.head.net.0.weight": {
          "dtype": "torch.float32",
          "sha256": "5fd29bad3a3457a72e6d200093e28c9a867878c990712604c5f4ab7d0033b771",
          "shape": [
            512,
            256
          ]
        },
        "model.head.net.3.bias": {
          "dtype": "torch.float32",
          "sha256": "1abe7888fce47d13cd83b088c1f3a00c2ab13bfe5a85e27bbfe60d78a8eb491c",
          "shape": [
            22
          ]
        },
        "model.head.net.3.weight": {
          "dtype": "torch.float32",
          "sha256": "d000e86ed823dcd2ee77ca57f9dc4bb2969dfbb2ffb5cc4549398faef8b292cc",
          "shape": [
            22,
            512
          ]
        }
      }
    },
    "files": {
      "emg": {
        "bytes": 13910102,
        "sha256": "4d53b5c6c548a16e29f466cd1917c2be62246adeb9cf0f730febcde3f71a5922"
      },
      "vision": {
        "bytes": 45881850,
        "sha256": "10be7acdcf011dde9bc9a8b8af5d93654c048a8bc2ef2335e8f9eaa5e2970e8a"
      }
    },
    "format": "rn18_fusion_complete_trained_initializers_v1",
    "full_paper_complete": false,
    "fusion_cohort_sha256": "1736318c9faf5740417ca676cd520d47ada84d16eac31f865e9cdf24a9f2e38c",
    "fusion_partition_content_sha256": "44e9ebc8a2f63a79699c5a976bf2717b0fe7a376427c7c46f538d0745c2b4bd0",
    "producer_training_verified": true,
    "protocol_sha256": "fa505397b72c600bea9987d8a96a41d324e9064cb1cf8b83c876867462362ac9",
    "scientific_eligible": false,
    "scope": "full",
    "sha256": "bfef9ec860477e2c643a372df9ea1d328a7c14523f188aa1304267b6b9d8d08d",
    "sources_sha256": {
      "egoemg_repro/fusion_initializer_handoff.py": "9fdf4842d12b61ed5092746b06a0284af4a2abca91c93a388c67fcda930b5f80",
      "egoemg_repro/fusion_literal_runtime.py": "3b357bdb0e41f3536c2296324917465ad25b2d13c8cccb553815fb729310ca60",
      "egoemg_repro/rn18_fusion_cohort.py": "0eca1ac9b5e46421b667d4a50539b14d4a9df6f70cce04d383d7b0f5a4ae225a",
      "egoemg_repro/rn18_fusion_data.py": "c9f90ced4afb7868f99510632a9daacb4b86fd7a81265d612468b77faa1c205d",
      "egoemg_repro/rn18_fusion_literal.py": "1160b90b66b8a9155c4ddad17e7f7cba1322d3f146b6da7011dc5539214fd5e7",
      "egoemg_repro/rn18_fusion_runtime.py": "c0b4d3ffaddef4b1b9c70146c7d21c085bdda257e2326074749bc8189de076ef",
      "egoemg_repro/rn18_fusion_trained_handoff.py": "54d957d58d1b28ebe827774a2df406545504a0e971b09099990f77032354bb41",
      "egoemg_repro/rn18_fusion_transfer.py": "01831c0bdc3974a90b215cf84d1e4f43d05082f3d0f71a4a896b6f90e811cf34",
      "egoemg_repro/rn18_fusion_vision_producer.py": "46080ff9dc1f491ecf1375109ec191204c38f032f2929cb4de62777f1fd10801",
      "egoemg_repro/rn18_initializer_cohort.py": "3a57e264fc316578c44aa813a6ae153854911373bb3b61edcf0750a8b8544be8",
      "egoemg_repro/rn18_initializer_data.py": "700afbc082841236792900d9856bbc26bca7c870dc47ccf4c22f78a2a58bccff",
      "egoemg_repro/rn18_initializer_train.py": "3f3270d2fe41eab05048ab2d271582e814cf18551cb432fa258947a7fefb501e"
    },
    "trained_branches_admitted": true,
    "vision": {
      "checkpoint_selection": "LAST",
      "current_input_binding": {
        "all_consumed_raw_bytes_match": true,
        "current_fusion_cohort_sha256": "1736318c9faf5740417ca676cd520d47ada84d16eac31f865e9cdf24a9f2e38c",
        "current_fusion_counts": {
          "test": 2792,
          "train": 62470,
          "val": 1052
        },
        "footprints": {
          "holdout_union_rows": {
            "test": 16752000,
            "val": 6312000
          },
          "overlap_rows": {
            "test": 0,
            "val": 0
          },
          "train_union_rows": 41715600
        },
        "future_fusion_partition_content_sha256": "44e9ebc8a2f63a79699c5a976bf2717b0fe7a376427c7c46f538d0745c2b4bd0",
        "legacy_only_wrist_valid_payloads": {
          "mocap_left_wrist_angles_valid": {
            "bytes": 66161725,
            "sha256": "d48e0430adac0b7ffd0e048e05cdd0fc084c90f28387ca79406dbf62bc9d9389"
          },
          "mocap_right_wrist_angles_valid": {
            "bytes": 66161725,
            "sha256": "6175793773e197b2cdb825d0b131ad686afeaeb1180b9852e0953b7fd986349d"
          }
        },
        "reconstructed_legacy_partition_sha256": "4a355040cac040e5fa4527879e862a57c3dd2e42f46e62e3eba838b65c009812",
        "shared_actual_payload_files": 53,
        "shared_actual_payloads_sha256": "273fa20686ce07d56025573a02d054bb6f2cf2ebffc6ed8a1ffa898bbbbc5897",
        "source_counts": {
          "test": 2792,
          "train": 63172,
          "val": 1052
        }
      },
      "current_raw_input_bytes_revalidated": true,
      "emg_initializer_admitted": false,
      "export_file_sha256": "10be7acdcf011dde9bc9a8b8af5d93654c048a8bc2ef2335e8f9eaa5e2970e8a",
      "format": "rn18_fusion_completed_legacy_vision_last_admission_v1",
      "full_paper_complete": false,
      "fusion_initializers_complete": false,
      "model_state_sha256": "a6718b66222e4fc9ea67dcb7200efdea48cf3058a54292c6528ba82c897fffdb",
      "producer": {
        "actual_history_counts": {
          "final_test": 1,
          "start": 1,
          "train_step": 10650,
          "validation": 150
        },
        "checkpoint_selection": "LAST",
        "completed_epochs": 150,
        "current_raw_input_bytes_revalidated": false,
        "format": "known_completed_legacy_rn18_last_reader_v1",
        "full_paper_complete": false,
        "fusion_initializers_complete": false,
        "historical_reader_isolated": true,
        "model_state_sha256": "a6718b66222e4fc9ea67dcb7200efdea48cf3058a54292c6528ba82c897fffdb",
        "native_optimizer_parameters": 64,
        "native_state_tensors": 124,
        "optimizer_executions": 10650,
        "producer_family": "legacy-resnet18-150-0.01-nojitter",
        "producer_training_verified": true,
        "reader_sources_sha256": {
          "egoemg_repro/__init__.py": "6103cfcd4a90339b804308dcb82a86448749c728fd093ebfab546ff9bfdb3e96",
          "egoemg_repro/audit.py": "86cf0c74086f2581f430ee88d8725df10c3389379535ef9036e2d7a892b27cdd",
          "egoemg_repro/crops.py": "1aa0871d6ede77d93988d455d0930e1f42470c1a92a47cfc8e49252bbfc0695e",
          "egoemg_repro/data.py": "de7a716c6616277f6189c94abb09803f950dd27bc1f9f7193b1a000758a88f0e",
          "egoemg_repro/download.py": "5063fc0a5a0dbb1929c054a9cac86d3f90c01b375171660b6d75e2707e684e75",
          "egoemg_repro/metrics.py": "12d6eb170997ae78ddbf5602d1252d05a778f1357f17b28aa3b96d06a618798e",
          "egoemg_repro/partitions.py": "ec816483e5d567cec45df77bd6823e97c201d3fd20f45b0d8004827565fdf678",
          "egoemg_repro/train.py": "f40f375d11bd535bd54edb4a6d98f399e3c74ea6c9f846ba94210a203e808418",
          "egoemg_repro/visual.py": "588a18cfd0bbd106c6843fbbe6e436635784cd2b0ea7b59f3db33878ab1f084b"
        },
        "scientific_eligible": false,
        "sha256": "766449b69e27ea68d4e441e2e79bef36be9a1092e776a0937f3f7497a3fe2a65",
        "source_adaptations": {
          "jitter": "disabled in legacy producer",
          "loss": "legacy single fingertip0.01, not pending standalone native0.02",
          "masks": "legacy separate wrist-valid/finite masking and nan_to_num",
          "normalization": "train-only per-hand/channel, not future fusion published vectors"
        },
        "source_checkpoint_sha256": "e9bf7c7fa2abee7997c2c7c50f750137ded4ae8dad1cba998f52392516db5fc0",
        "source_normalization_sha256": "bdfba74af8f084160a770732767d1081bb949b9d6dcf2d0cb1b1cecc20f7e827",
        "source_partition_content_sha256": "4a355040cac040e5fa4527879e862a57c3dd2e42f46e62e3eba838b65c009812",
        "source_recipe": {
          "augmentation": "none",
          "batch_size": 900,
          "epochs": 150,
          "eta_min": 1e-05,
          "eval_stride": 12000,
          "fingertip_weight": 0.01,
          "lr": 0.0001,
          "num_workers": 4,
          "optimizer_class": "AdamW",
          "precision": "bfloat16",
          "seed": 42,
          "train_stride": 1200,
          "val_fraction": 0.12,
          "weight_decay": 0.01,
          "window_length": 12000
        },
        "source_reproduction_sha256": "aee6c1be984e7486548e8c7da4aee083ed205b43582980b18f19e7e2dfe72af8",
        "source_run_files_sha256": {
          "config.json": "2ab2c4ab20e58fdb71db7ac1606b6b2b127ee507d872dd42782b8cde38f9bac8",
          "environment.json": "093227449f34c2e572f87aa9efa2d15a265d1ca6d0e3bacef974f6acae3a6bdb",
          "last.pt": "e9bf7c7fa2abee7997c2c7c50f750137ded4ae8dad1cba998f52392516db5fc0",
          "normalization.json": "62734cd4766804b1e405ad0d64c172b9b6a9fe367988c6310ab5d4652b341f49",
          "official_module.json": "31a1509528e9370f948ac81d1e7ee76d6d5282ed1b67b142a201103f21c668ce",
          "partitions.json": "534a20d8710b0706362a6b26e02c7c4e06bf854e00fb9e073a8179ecc048d896",
          "progress.jsonl": "de43369906460ebf5ad2c4aca3acd4ddba3de01b48135b34d41b6098ee7e7736",
          "provenance.json": "ae260b37ea588f188d26619c54ca2123a5c7d01de652b56b4d5057c02e4bd871",
          "recipe_source.json": "ad3d43fd4757408f8357cb15769fe314ec1ef41913e03fdc3c1566b5a2ebf33c",
          "run.json": "2f81eab3d0ce73a804583ca8f854ffc26e80d779678046e551838aeb7baffa17",
          "summary.json": "8014c41af5dcf7598549f52d1bdfaefa5a09ee528a61b3eb8821e74ca16cd568"
        },
        "source_runtime_files_sha256": {
          "egoemg_repro/__init__.py": "6103cfcd4a90339b804308dcb82a86448749c728fd093ebfab546ff9bfdb3e96",
          "egoemg_repro/crops.py": "1aa0871d6ede77d93988d455d0930e1f42470c1a92a47cfc8e49252bbfc0695e",
          "egoemg_repro/data.py": "de7a716c6616277f6189c94abb09803f950dd27bc1f9f7193b1a000758a88f0e",
          "egoemg_repro/download.py": "5063fc0a5a0dbb1929c054a9cac86d3f90c01b375171660b6d75e2707e684e75",
          "egoemg_repro/metrics.py": "12d6eb170997ae78ddbf5602d1252d05a778f1357f17b28aa3b96d06a618798e",
          "egoemg_repro/partitions.py": "ec816483e5d567cec45df77bd6823e97c201d3fd20f45b0d8004827565fdf678",
          "egoemg_repro/train.py": "f40f375d11bd535bd54edb4a6d98f399e3c74ea6c9f846ba94210a203e808418",
          "egoemg_repro/visual.py": "588a18cfd0bbd106c6843fbbe6e436635784cd2b0ea7b59f3db33878ab1f084b"
        },
        "source_validation_selected_step": 6319,
        "sources_sha256": {
          "egoemg_repro/__init__.py": "6103cfcd4a90339b804308dcb82a86448749c728fd093ebfab546ff9bfdb3e96",
          "egoemg_repro/audit.py": "86cf0c74086f2581f430ee88d8725df10c3389379535ef9036e2d7a892b27cdd",
          "egoemg_repro/crops.py": "1aa0871d6ede77d93988d455d0930e1f42470c1a92a47cfc8e49252bbfc0695e",
          "egoemg_repro/data.py": "de7a716c6616277f6189c94abb09803f950dd27bc1f9f7193b1a000758a88f0e",
          "egoemg_repro/download.py": "5063fc0a5a0dbb1929c054a9cac86d3f90c01b375171660b6d75e2707e684e75",
          "egoemg_repro/fusion_initializer_handoff.py": "9fdf4842d12b61ed5092746b06a0284af4a2abca91c93a388c67fcda930b5f80",
          "egoemg_repro/metrics.py": "12d6eb170997ae78ddbf5602d1252d05a778f1357f17b28aa3b96d06a618798e",
          "egoemg_repro/partitions.py": "ec816483e5d567cec45df77bd6823e97c201d3fd20f45b0d8004827565fdf678",
          "egoemg_repro/rn18_fusion_cohort.py": "0eca1ac9b5e46421b667d4a50539b14d4a9df6f70cce04d383d7b0f5a4ae225a",
          "egoemg_repro/rn18_fusion_data.py": "c9f90ced4afb7868f99510632a9daacb4b86fd7a81265d612468b77faa1c205d",
          "egoemg_repro/rn18_fusion_literal.py": "1160b90b66b8a9155c4ddad17e7f7cba1322d3f146b6da7011dc5539214fd5e7",
          "egoemg_repro/rn18_fusion_vision_producer.py": "46080ff9dc1f491ecf1375109ec191204c38f032f2929cb4de62777f1fd10801",
          "egoemg_repro/train.py": "10696073b661823a9bf47ddda26b4c3b27324b00c9b62e3841a670409e872b6a",
          "egoemg_repro/vision_literal_runtime.py": "bb430267f819c1a1c46ed961397a81c3b9619d9a48f321939cc78808bc85e4d0",
          "egoemg_repro/visual.py": "b8c390c0291abd67ee58a6f46caddce38296f2013b9f19088c330792b3d76cdb"
        },
        "tensor_inventory": {
          "backbone.0.weight": {
            "dtype": "torch.float32",
            "sha256": "213d57dffee38f02e590f556453243f865a689d38de01daf299cb86545cd29fa",
            "shape": [
              64,
              3,
              7,
              7
            ]
          },
          "backbone.1.bias": {
            "dtype": "torch.float32",
            "sha256": "5cf09c4d1b8c18ba2a7b1c7766dc682339bd61e0d270b64a99efa171dadb8aef",
            "shape": [
              64
            ]
          },
          "backbone.1.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "1c7bf994506d7fce9e3b80f32227f3b25a5b681c0ef8337965963d08631c58a4",
            "shape": []
          },
          "backbone.1.running_mean": {
            "dtype": "torch.float32",
            "sha256": "a58fb1fe496b8e1e0dad5eecf71302acf6076445b9f165fe02e1c092d4fe3f6a",
            "shape": [
              64
            ]
          },
          "backbone.1.running_var": {
            "dtype": "torch.float32",
            "sha256": "ad3717a7c55d411b56b372a786310052a3c563cc9d27f8b77cfbd646e8e0b5ce",
            "shape": [
              64
            ]
          },
          "backbone.1.weight": {
            "dtype": "torch.float32",
            "sha256": "a376cd5dd5b2f95c5f4393a1d3bb11f791736c9271d6f9c5596f7cbf5f63def3",
            "shape": [
              64
            ]
          },
          "backbone.4.0.bn1.bias": {
            "dtype": "torch.float32",
            "sha256": "bb9e9d99420778da95e3acb1a3db8c99170da7f43f964de4cd3d74e38cada817",
            "shape": [
              64
            ]
          },
          "backbone.4.0.bn1.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "8468d9b503180a92d8fcca731b8a00242788f5e7cb15e81d57355e39a8ed343e",
            "shape": []
          },
          "backbone.4.0.bn1.running_mean": {
            "dtype": "torch.float32",
            "sha256": "327b00215d946d37d8ce2bd5acf75608062dd9ce926b617868b3aba22dde8143",
            "shape": [
              64
            ]
          },
          "backbone.4.0.bn1.running_var": {
            "dtype": "torch.float32",
            "sha256": "b3945015fb7cc534e0b43617dbf9f8f00f248112ad10b2dafe85da704135c2b6",
            "shape": [
              64
            ]
          },
          "backbone.4.0.bn1.weight": {
            "dtype": "torch.float32",
            "sha256": "d74d4ca560135641d0ded647a7dd6058ee7507a88315d056336601ecb1b237d6",
            "shape": [
              64
            ]
          },
          "backbone.4.0.bn2.bias": {
            "dtype": "torch.float32",
            "sha256": "53ae11ec4955036ed62378379aa9d2c73cc2666d89aa62095d7d64f7d3f2a718",
            "shape": [
              64
            ]
          },
          "backbone.4.0.bn2.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "7f83a3bc340856fee649324560c1bfc47d6407a38d964afccc70e59f270431e5",
            "shape": []
          },
          "backbone.4.0.bn2.running_mean": {
            "dtype": "torch.float32",
            "sha256": "c63a3a041a6b8563965b6d5972fe20282ade8d8686f8016d94561f64779b2358",
            "shape": [
              64
            ]
          },
          "backbone.4.0.bn2.running_var": {
            "dtype": "torch.float32",
            "sha256": "bcf5b848e8c6607eed43be8beaca41b60c64d541eee4fb618a15004bf870c3ea",
            "shape": [
              64
            ]
          },
          "backbone.4.0.bn2.weight": {
            "dtype": "torch.float32",
            "sha256": "14eb0f47c434f024c2307d673c7655be8df88dd135176267464e5140c474be98",
            "shape": [
              64
            ]
          },
          "backbone.4.0.conv1.weight": {
            "dtype": "torch.float32",
            "sha256": "7db120299002b6c85933b5b9dd64fb76ba5667e21c24b4e41afb6948c3e17393",
            "shape": [
              64,
              64,
              3,
              3
            ]
          },
          "backbone.4.0.conv2.weight": {
            "dtype": "torch.float32",
            "sha256": "ccfefeda1ab21d38719fca4be92c0e735e5d0fdec7904e986fe208bb4850d4c2",
            "shape": [
              64,
              64,
              3,
              3
            ]
          },
          "backbone.4.1.bn1.bias": {
            "dtype": "torch.float32",
            "sha256": "6e673c56b2afcb684a2b5cb7fed873505127401a42bb68191b5ec2ae0aafd568",
            "shape": [
              64
            ]
          },
          "backbone.4.1.bn1.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "52057987cc8bcbf065c3da183010042c45863ad8fbd8970d2bf14c23acf0f7e1",
            "shape": []
          },
          "backbone.4.1.bn1.running_mean": {
            "dtype": "torch.float32",
            "sha256": "20a7d4fc7735cd0188faff20e88b23146484806bed5aa5c8461d7062b82b4ad3",
            "shape": [
              64
            ]
          },
          "backbone.4.1.bn1.running_var": {
            "dtype": "torch.float32",
            "sha256": "512edbdc3d27ebbd75b75ca7139492213def764ee956932277c90770d87d9f79",
            "shape": [
              64
            ]
          },
          "backbone.4.1.bn1.weight": {
            "dtype": "torch.float32",
            "sha256": "d064decfbdbed5b42318e694f1418fdc638fa75cdb5eb774e2b64c97442c9a5b",
            "shape": [
              64
            ]
          },
          "backbone.4.1.bn2.bias": {
            "dtype": "torch.float32",
            "sha256": "547907d6fc4da7a4de8ea2d26a3d21f6e16c07a959e773dd34dfb520cb0278b7",
            "shape": [
              64
            ]
          },
          "backbone.4.1.bn2.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "2dcaeb3be4994379778f0266dda7b938351c95ec7a1c2c4ef2a9dbbec51ab248",
            "shape": []
          },
          "backbone.4.1.bn2.running_mean": {
            "dtype": "torch.float32",
            "sha256": "80b6ac5ecbc0c880e96e03f57bcb8c243c55cb455ff483f4770d4502cba73809",
            "shape": [
              64
            ]
          },
          "backbone.4.1.bn2.running_var": {
            "dtype": "torch.float32",
            "sha256": "37ed2dbfb37b43c467a5bf9f6a148b2b6e778172f067dc7f0964c7e19420c3ce",
            "shape": [
              64
            ]
          },
          "backbone.4.1.bn2.weight": {
            "dtype": "torch.float32",
            "sha256": "eb72b3dd3fc11839679494fa997f175889a06391d882becf9373f00191008498",
            "shape": [
              64
            ]
          },
          "backbone.4.1.conv1.weight": {
            "dtype": "torch.float32",
            "sha256": "2db8f3472b2fa06bd09b6c55f4429cb49560ac59b69ca9317089f4df719d22f1",
            "shape": [
              64,
              64,
              3,
              3
            ]
          },
          "backbone.4.1.conv2.weight": {
            "dtype": "torch.float32",
            "sha256": "d36db7743c048da8ea7ab8347e94e608d3d5c8ca0bd515e2d3df1adaff6e39e9",
            "shape": [
              64,
              64,
              3,
              3
            ]
          },
          "backbone.5.0.bn1.bias": {
            "dtype": "torch.float32",
            "sha256": "cbba26ca2497d36449650dea14c3388e40454f4507726a3ab54d8818366ca634",
            "shape": [
              128
            ]
          },
          "backbone.5.0.bn1.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "4aa8ecfb9013c18c61343a46bb73dfe019bb284d47c57e019f9d4eae4390f901",
            "shape": []
          },
          "backbone.5.0.bn1.running_mean": {
            "dtype": "torch.float32",
            "sha256": "9dd97c1641b72a9693978247447582e0a18f6066662dab12743229a97652bba4",
            "shape": [
              128
            ]
          },
          "backbone.5.0.bn1.running_var": {
            "dtype": "torch.float32",
            "sha256": "9744c0e015d6a2ee55b566e85fd1b90e89243270780d41933fe052c035d45fdc",
            "shape": [
              128
            ]
          },
          "backbone.5.0.bn1.weight": {
            "dtype": "torch.float32",
            "sha256": "26d4d06d040950784031e0c65def6fdabca5b0b46cb3d8ac79bd1b4a7c73a760",
            "shape": [
              128
            ]
          },
          "backbone.5.0.bn2.bias": {
            "dtype": "torch.float32",
            "sha256": "69fb514943bc04003517ab7cc5cf01a9a17c2cfa85db210a9bcb0d09b52010dd",
            "shape": [
              128
            ]
          },
          "backbone.5.0.bn2.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "d1836d3a02835cac75e72d71ea7a9cc5f2d2be5c7502e957e60b3f4f8311f4a0",
            "shape": []
          },
          "backbone.5.0.bn2.running_mean": {
            "dtype": "torch.float32",
            "sha256": "86a84c681d66747b3a1f278c638216f0fd7f431b8b98a05233a55ad5a4f6a702",
            "shape": [
              128
            ]
          },
          "backbone.5.0.bn2.running_var": {
            "dtype": "torch.float32",
            "sha256": "ff145fe3827fcc9d2df6f4e2a82592c12c2e6001516b25d9de49236c79c823b3",
            "shape": [
              128
            ]
          },
          "backbone.5.0.bn2.weight": {
            "dtype": "torch.float32",
            "sha256": "6ce3f8c5697e355ae4192ac4cda522200dca4e4e15325f72ecb6e51a4db59a51",
            "shape": [
              128
            ]
          },
          "backbone.5.0.conv1.weight": {
            "dtype": "torch.float32",
            "sha256": "e83c079ef90003940c59a52650aca03f1f3e327addb5fcd169845063efa11692",
            "shape": [
              128,
              64,
              3,
              3
            ]
          },
          "backbone.5.0.conv2.weight": {
            "dtype": "torch.float32",
            "sha256": "c1679c25f8787015d9708f127f119a2a2c252f4c6a76d33544fd6322ec5c2d43",
            "shape": [
              128,
              128,
              3,
              3
            ]
          },
          "backbone.5.0.downsample.0.weight": {
            "dtype": "torch.float32",
            "sha256": "a3dc1acd00af81df88da6ad3db2a54626bb091e64efaf9f5757cc3b26229a3e0",
            "shape": [
              128,
              64,
              1,
              1
            ]
          },
          "backbone.5.0.downsample.1.bias": {
            "dtype": "torch.float32",
            "sha256": "a037465e2165eb306c4d2eb6119cb6a0186698c0e36b21d7b651cebb907c6132",
            "shape": [
              128
            ]
          },
          "backbone.5.0.downsample.1.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "53f8d321fa83834e1affb8fceaf717101ebd092735b3d424e9cc50507078c76d",
            "shape": []
          },
          "backbone.5.0.downsample.1.running_mean": {
            "dtype": "torch.float32",
            "sha256": "0e2872641bccb61a6fbbfc33b30e4c7430a68df6b3407f29a36be8eae049ea7d",
            "shape": [
              128
            ]
          },
          "backbone.5.0.downsample.1.running_var": {
            "dtype": "torch.float32",
            "sha256": "b1dc46a23262d18bb67055b7138fd01f12194be88ec4f556ed19542ad5046f32",
            "shape": [
              128
            ]
          },
          "backbone.5.0.downsample.1.weight": {
            "dtype": "torch.float32",
            "sha256": "06c89ff14cfddf2699e1a60626038eddbf94c2023b4d4806f28325d7affd7df6",
            "shape": [
              128
            ]
          },
          "backbone.5.1.bn1.bias": {
            "dtype": "torch.float32",
            "sha256": "aa4d9876fb3ae2e90e5b01744f3cdab14248bd7d4d8ab405d20e6071679365bc",
            "shape": [
              128
            ]
          },
          "backbone.5.1.bn1.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "51947480dfe8b580e06e3045f88e036d21c1ef6b961133610dcbfa315a18490e",
            "shape": []
          },
          "backbone.5.1.bn1.running_mean": {
            "dtype": "torch.float32",
            "sha256": "cfd4fff3962a107bc664a07485f95da20c44c0cb6abd42c285c8a91b7ed3e30e",
            "shape": [
              128
            ]
          },
          "backbone.5.1.bn1.running_var": {
            "dtype": "torch.float32",
            "sha256": "374501556362ff7787f27d2577f272de2282dd77b32206ce2b783cb6cc8ca85b",
            "shape": [
              128
            ]
          },
          "backbone.5.1.bn1.weight": {
            "dtype": "torch.float32",
            "sha256": "3d5d8a23cf662f68138f6119d207ea138fc3948ddc6bf918504a2d1fa88b1264",
            "shape": [
              128
            ]
          },
          "backbone.5.1.bn2.bias": {
            "dtype": "torch.float32",
            "sha256": "b24392a2415a8a5dc4f05ac412fa64165e551570e34a270b611f05a37c735822",
            "shape": [
              128
            ]
          },
          "backbone.5.1.bn2.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "f0a94a283594e7967495eebe67bccdddbd0453b2029b34c63d4532bc16730182",
            "shape": []
          },
          "backbone.5.1.bn2.running_mean": {
            "dtype": "torch.float32",
            "sha256": "f57022f4f188f99882b668da1c0b8111fc2961b4ccf7ed4e3186fcac8c5a90e5",
            "shape": [
              128
            ]
          },
          "backbone.5.1.bn2.running_var": {
            "dtype": "torch.float32",
            "sha256": "c4fc0bef3d533b60a88fc6c654bc94e7db96e5fbde04b7394faf5575b86ef77a",
            "shape": [
              128
            ]
          },
          "backbone.5.1.bn2.weight": {
            "dtype": "torch.float32",
            "sha256": "fcfeb4f8c240eeafa74ad0b0a51d01c20229b6cedd52c4049702b0cf83e87fa7",
            "shape": [
              128
            ]
          },
          "backbone.5.1.conv1.weight": {
            "dtype": "torch.float32",
            "sha256": "f709b6105bf828ece64aee25a9bf028876a8e8002accdc31d31088cb15c6cf62",
            "shape": [
              128,
              128,
              3,
              3
            ]
          },
          "backbone.5.1.conv2.weight": {
            "dtype": "torch.float32",
            "sha256": "98da087c626cf79172c69f3eb65478f51a9b7e2110df388bed79a5bf9573aed8",
            "shape": [
              128,
              128,
              3,
              3
            ]
          },
          "backbone.6.0.bn1.bias": {
            "dtype": "torch.float32",
            "sha256": "2eb41c5a79e9f22abed64aa221de242b894681f602bbf1aaa6cb55419407f200",
            "shape": [
              256
            ]
          },
          "backbone.6.0.bn1.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "7f08900e5d527018a41bb6aaeee5da46b596c6f7796c6568d46fed5700214dbd",
            "shape": []
          },
          "backbone.6.0.bn1.running_mean": {
            "dtype": "torch.float32",
            "sha256": "da493ca60fa65d10547692d449f8fa6c9b7c01dd7c6f18a7114859b4f0c14b9d",
            "shape": [
              256
            ]
          },
          "backbone.6.0.bn1.running_var": {
            "dtype": "torch.float32",
            "sha256": "aa5d34e77d8f9de91bc6d8b0d9f46fe55c6648ab5dc87631b0783351eb288a67",
            "shape": [
              256
            ]
          },
          "backbone.6.0.bn1.weight": {
            "dtype": "torch.float32",
            "sha256": "35f94b7361cce5e33026add263d03d7e79ad44af905fd7dc2f2835534195af03",
            "shape": [
              256
            ]
          },
          "backbone.6.0.bn2.bias": {
            "dtype": "torch.float32",
            "sha256": "c01d1cb22ae8fc5595f59f93b3727e68ed275197abfa191a0aea997b450545fa",
            "shape": [
              256
            ]
          },
          "backbone.6.0.bn2.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "88d3d77ebc7349857e9d2fcf735878ea8ea1452ce8e5bf32d06239a32c77c327",
            "shape": []
          },
          "backbone.6.0.bn2.running_mean": {
            "dtype": "torch.float32",
            "sha256": "27d84e94032ce2ce5c31ecb07ad73cc43ff816d6fe12abe62d53bd33fb8b09b1",
            "shape": [
              256
            ]
          },
          "backbone.6.0.bn2.running_var": {
            "dtype": "torch.float32",
            "sha256": "cfb6b8f6fd0d837dbef4f03ae2a2327eae502348ea6f24e9a2cc57b5ea9419b2",
            "shape": [
              256
            ]
          },
          "backbone.6.0.bn2.weight": {
            "dtype": "torch.float32",
            "sha256": "ed7fa05bcba8183608653a5e8106c8eca15df7546c8cb04cda6f631fc77d9d08",
            "shape": [
              256
            ]
          },
          "backbone.6.0.conv1.weight": {
            "dtype": "torch.float32",
            "sha256": "efda9f25ee7d6cc4b8d7385f59b80b31edd9a2c3c87130a35905668490e34e6a",
            "shape": [
              256,
              128,
              3,
              3
            ]
          },
          "backbone.6.0.conv2.weight": {
            "dtype": "torch.float32",
            "sha256": "fa2353a38af33a0df5333c73936292a1ae96186794035a889f52a939089e577e",
            "shape": [
              256,
              256,
              3,
              3
            ]
          },
          "backbone.6.0.downsample.0.weight": {
            "dtype": "torch.float32",
            "sha256": "1272d07e372dc6fe5d6995bfd5007622a4300b6f1130a9ff44128f9f016d0d4f",
            "shape": [
              256,
              128,
              1,
              1
            ]
          },
          "backbone.6.0.downsample.1.bias": {
            "dtype": "torch.float32",
            "sha256": "5eef383b5062433c58d85ed03da6038e02cc0cd1c0588d261aeaee1e13c6cc02",
            "shape": [
              256
            ]
          },
          "backbone.6.0.downsample.1.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "d3d34a4f7b0dff5005e74f015cbde986b00b7d2bb628a625e9e598205c1db43e",
            "shape": []
          },
          "backbone.6.0.downsample.1.running_mean": {
            "dtype": "torch.float32",
            "sha256": "a673000dcf0fd3c61712bad7b083b61e5810624514e55f145a4a3ef6899f8f49",
            "shape": [
              256
            ]
          },
          "backbone.6.0.downsample.1.running_var": {
            "dtype": "torch.float32",
            "sha256": "80ff2baa162c8b34397940eace1fa30670b705eac3d34233288dab38725f57df",
            "shape": [
              256
            ]
          },
          "backbone.6.0.downsample.1.weight": {
            "dtype": "torch.float32",
            "sha256": "82f232462457fe95f29de8dc57e8bbca86c4bf30a2c566d795630e9f6d2ab2ea",
            "shape": [
              256
            ]
          },
          "backbone.6.1.bn1.bias": {
            "dtype": "torch.float32",
            "sha256": "302f1656b4305ea9adf03d63cbabff022a0a0d4e3f551f2002cf4652a2467a8a",
            "shape": [
              256
            ]
          },
          "backbone.6.1.bn1.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "cb8942fc0959fa47bcdb1e75b5d5252881b33857bbbb736230b662c0733d98a4",
            "shape": []
          },
          "backbone.6.1.bn1.running_mean": {
            "dtype": "torch.float32",
            "sha256": "3c840f74bc4879cda9f62a05bcb0b7ffc1558a8c4416ae1656ccc4b85d3cc15c",
            "shape": [
              256
            ]
          },
          "backbone.6.1.bn1.running_var": {
            "dtype": "torch.float32",
            "sha256": "64ecacd4da638440ab99b3e970895d98ed9d2bf4a88fec7612107a86c73ddadc",
            "shape": [
              256
            ]
          },
          "backbone.6.1.bn1.weight": {
            "dtype": "torch.float32",
            "sha256": "011763d7bd99121db61374c09150a4c3fc02508dd8cf820bc969b9bbd554070c",
            "shape": [
              256
            ]
          },
          "backbone.6.1.bn2.bias": {
            "dtype": "torch.float32",
            "sha256": "7b3ebc02cfdf579f22cfdca7e62ac2f2584710df23aa6f83777b1da5dd023ac7",
            "shape": [
              256
            ]
          },
          "backbone.6.1.bn2.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "a8bade65ba936991d3edfa4c2fa02a63033fba4e6c8484b62af8cc156363f67a",
            "shape": []
          },
          "backbone.6.1.bn2.running_mean": {
            "dtype": "torch.float32",
            "sha256": "fd5b2494e0f12c02431d3b153e1dd3e2c208ef1670a68b7a33f11714ce5e032c",
            "shape": [
              256
            ]
          },
          "backbone.6.1.bn2.running_var": {
            "dtype": "torch.float32",
            "sha256": "4c14128ae1730b9284d4e0c031cb210298b91951da13c5a4e921f3c4072f4042",
            "shape": [
              256
            ]
          },
          "backbone.6.1.bn2.weight": {
            "dtype": "torch.float32",
            "sha256": "4dfc71e2acf2ba1819cb8b07d82788b21d2ef556a908183c8abd1ba4936732e9",
            "shape": [
              256
            ]
          },
          "backbone.6.1.conv1.weight": {
            "dtype": "torch.float32",
            "sha256": "a1a556df6ad6ca9d46cbbbe4bd8f213dda77d017e3e1e313b841674fe722e4de",
            "shape": [
              256,
              256,
              3,
              3
            ]
          },
          "backbone.6.1.conv2.weight": {
            "dtype": "torch.float32",
            "sha256": "d492515f46a92739058f89938fe52d69975957e459fcfe56e53e3d7a98e14dfd",
            "shape": [
              256,
              256,
              3,
              3
            ]
          },
          "backbone.7.0.bn1.bias": {
            "dtype": "torch.float32",
            "sha256": "17f490f980432c68b02f156e01f592563bdbdc98638b5f09f97dbc2d38888539",
            "shape": [
              512
            ]
          },
          "backbone.7.0.bn1.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "9687f952409405079931d9e2e05d4dff69fbaa3b4f83c15d70f5794a2a804f10",
            "shape": []
          },
          "backbone.7.0.bn1.running_mean": {
            "dtype": "torch.float32",
            "sha256": "5f2df200455959b74aa498a9a3fe878fb0133ce774cc925c4069d01cd544e70c",
            "shape": [
              512
            ]
          },
          "backbone.7.0.bn1.running_var": {
            "dtype": "torch.float32",
            "sha256": "6ad0445bb7fd0326d54ac42f7825e2cf1fcb6b064f016813328aadcdb6ee4681",
            "shape": [
              512
            ]
          },
          "backbone.7.0.bn1.weight": {
            "dtype": "torch.float32",
            "sha256": "802511d6c9b40da98467cbd165a5f2760e80f1ddab5be1f8fd4ca5fd71173a23",
            "shape": [
              512
            ]
          },
          "backbone.7.0.bn2.bias": {
            "dtype": "torch.float32",
            "sha256": "5dab38917de6fba127c7827dc4beac8621284653143c04ba089a61d1eaef44c0",
            "shape": [
              512
            ]
          },
          "backbone.7.0.bn2.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "0883bc45e650d3b125176aef6770b4ed319f188c08a290ad77d08f49b1f2a960",
            "shape": []
          },
          "backbone.7.0.bn2.running_mean": {
            "dtype": "torch.float32",
            "sha256": "c2125e3c169f4080aa83eab54dc786f736ba9d6aeb198079b70e4e817646c923",
            "shape": [
              512
            ]
          },
          "backbone.7.0.bn2.running_var": {
            "dtype": "torch.float32",
            "sha256": "69c2934f37bbf72681b9df975ea55801724f5c10eb2141a92abf3cb356622d03",
            "shape": [
              512
            ]
          },
          "backbone.7.0.bn2.weight": {
            "dtype": "torch.float32",
            "sha256": "ce349a4923d5a7162a7398ae88ab325d18ec52051d70d9feab31032f8f6ce1d0",
            "shape": [
              512
            ]
          },
          "backbone.7.0.conv1.weight": {
            "dtype": "torch.float32",
            "sha256": "3893decedd5c7f61cf10438ea5d943312d502000b45513c285b7c95972774dad",
            "shape": [
              512,
              256,
              3,
              3
            ]
          },
          "backbone.7.0.conv2.weight": {
            "dtype": "torch.float32",
            "sha256": "03c12f381543f70ae9c13d1c023bbe42a57875bcbba6625a359ddd4222a517de",
            "shape": [
              512,
              512,
              3,
              3
            ]
          },
          "backbone.7.0.downsample.0.weight": {
            "dtype": "torch.float32",
            "sha256": "0da89e8c01b157e256d49e4c3042aa4d312791a55a8ebc05078b2e0933b07864",
            "shape": [
              512,
              256,
              1,
              1
            ]
          },
          "backbone.7.0.downsample.1.bias": {
            "dtype": "torch.float32",
            "sha256": "b9f46ffbc511bf1a327e52be175378128d221a125780c8dc400fc57790a7c133",
            "shape": [
              512
            ]
          },
          "backbone.7.0.downsample.1.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "6a552e8ec6de1eb04998e9b299b161585287e0e2ea6d73df729807129391b564",
            "shape": []
          },
          "backbone.7.0.downsample.1.running_mean": {
            "dtype": "torch.float32",
            "sha256": "5aca79c2487584657bfab200b796cfd068007c4e2a1249816cf990e717cd1b11",
            "shape": [
              512
            ]
          },
          "backbone.7.0.downsample.1.running_var": {
            "dtype": "torch.float32",
            "sha256": "2bd5a613c29921f7c77594a053cb9029004cf5abb82f8d2d54604c8941ac7fee",
            "shape": [
              512
            ]
          },
          "backbone.7.0.downsample.1.weight": {
            "dtype": "torch.float32",
            "sha256": "97630cb1015b4d609c21e54a6633cf397fe777538de928e26609a3c1f9317898",
            "shape": [
              512
            ]
          },
          "backbone.7.1.bn1.bias": {
            "dtype": "torch.float32",
            "sha256": "ed7c44b78b3a363bdd08658e94debb691a14011376c1346744c95ad8a4bfd1d8",
            "shape": [
              512
            ]
          },
          "backbone.7.1.bn1.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "954cab30d6a794175ef4d8380ce73fc07aa30d03ce3f18097e0ca82aedd32d2a",
            "shape": []
          },
          "backbone.7.1.bn1.running_mean": {
            "dtype": "torch.float32",
            "sha256": "d3e4a5e85a331b98a9f82021109619b7653f41f4f565e142d5e63101daa0cc94",
            "shape": [
              512
            ]
          },
          "backbone.7.1.bn1.running_var": {
            "dtype": "torch.float32",
            "sha256": "c1b176906d1fd06b7dbddc14a200af997767f985af30c024bd7890b674500488",
            "shape": [
              512
            ]
          },
          "backbone.7.1.bn1.weight": {
            "dtype": "torch.float32",
            "sha256": "d9e542fd33270b602bb7df9654e42e87c20a1c1306d830506deca510c6acfc46",
            "shape": [
              512
            ]
          },
          "backbone.7.1.bn2.bias": {
            "dtype": "torch.float32",
            "sha256": "309afa4a5e4f9e536522ab0187d43a9367bebfb80f0309c3ab259baff4ab2319",
            "shape": [
              512
            ]
          },
          "backbone.7.1.bn2.num_batches_tracked": {
            "dtype": "torch.int64",
            "sha256": "840b9d8b3f7824169b19fb2cce1dfd68e161b887b26366e820f5374a02924c81",
            "shape": []
          },
          "backbone.7.1.bn2.running_mean": {
            "dtype": "torch.float32",
            "sha256": "e34e631b7fd2ee5c04bf87d16cd65d3ed0028415d34d0ffc031aea371699affd",
            "shape": [
              512
            ]
          },
          "backbone.7.1.bn2.running_var": {
            "dtype": "torch.float32",
            "sha256": "9d949fbf558f755fc68105dfb70f1d639d1ce745cc80de858f5ee3214029cb0e",
            "shape": [
              512
            ]
          },
          "backbone.7.1.bn2.weight": {
            "dtype": "torch.float32",
            "sha256": "0bd80284ee3f52ab07739638a91378d5f1c2d2a77a8b4d73e60e9a4397a6defd",
            "shape": [
              512
            ]
          },
          "backbone.7.1.conv1.weight": {
            "dtype": "torch.float32",
            "sha256": "d868de32e44fe81f8c2431f77d5a3e15445f37287f736c8a0db6f2300f0b2f4d",
            "shape": [
              512,
              512,
              3,
              3
            ]
          },
          "backbone.7.1.conv2.weight": {
            "dtype": "torch.float32",
            "sha256": "a4a262c16dfe57b10c3a3bf80e1938f6380bc735ae4267455744f22df9f3f759",
            "shape": [
              512,
              512,
              3,
              3
            ]
          },
          "head.0.bias": {
            "dtype": "torch.float32",
            "sha256": "83b8712393eb60580f9e2adeb7f43fd3b28e3b43060126fa0e3339e72187b85a",
            "shape": [
              512
            ]
          },
          "head.0.weight": {
            "dtype": "torch.float32",
            "sha256": "748c4228f31bae47d4f186fc0b87d08dad0632e09ccd0845ac640b1b0bb7227b",
            "shape": [
              512,
              512
            ]
          },
          "head.3.bias": {
            "dtype": "torch.float32",
            "sha256": "8f74b520c48e94d019f6e7ee7145817b8f0bee4a4a73e7b43b771d35522e253c",
            "shape": [
              22
            ]
          },
          "head.3.weight": {
            "dtype": "torch.float32",
            "sha256": "eb441fe340079f7fff4e5328b78e77762066e6e6628ce77f77f470a151ebe48a",
            "shape": [
              22,
              512
            ]
          }
        }
      },
      "producer_training_verified": true,
      "scientific_eligible": false,
      "scope": "completed vision branch only",
      "sha256": "5f1b1f4bfe39d07a6b71f7e13a544b1fbdaba78871afe9242be174a36620df41",
      "sources_sha256": {
        "egoemg_repro/__init__.py": "6103cfcd4a90339b804308dcb82a86448749c728fd093ebfab546ff9bfdb3e96",
        "egoemg_repro/audit.py": "86cf0c74086f2581f430ee88d8725df10c3389379535ef9036e2d7a892b27cdd",
        "egoemg_repro/crops.py": "1aa0871d6ede77d93988d455d0930e1f42470c1a92a47cfc8e49252bbfc0695e",
        "egoemg_repro/data.py": "de7a716c6616277f6189c94abb09803f950dd27bc1f9f7193b1a000758a88f0e",
        "egoemg_repro/download.py": "5063fc0a5a0dbb1929c054a9cac86d3f90c01b375171660b6d75e2707e684e75",
        "egoemg_repro/fusion_initializer_handoff.py": "9fdf4842d12b61ed5092746b06a0284af4a2abca91c93a388c67fcda930b5f80",
        "egoemg_repro/metrics.py": "12d6eb170997ae78ddbf5602d1252d05a778f1357f17b28aa3b96d06a618798e",
        "egoemg_repro/partitions.py": "ec816483e5d567cec45df77bd6823e97c201d3fd20f45b0d8004827565fdf678",
        "egoemg_repro/rn18_fusion_cohort.py": "0eca1ac9b5e46421b667d4a50539b14d4a9df6f70cce04d383d7b0f5a4ae225a",
        "egoemg_repro/rn18_fusion_data.py": "c9f90ced4afb7868f99510632a9daacb4b86fd7a81265d612468b77faa1c205d",
        "egoemg_repro/rn18_fusion_literal.py": "1160b90b66b8a9155c4ddad17e7f7cba1322d3f146b6da7011dc5539214fd5e7",
        "egoemg_repro/rn18_fusion_vision_producer.py": "46080ff9dc1f491ecf1375109ec191204c38f032f2929cb4de62777f1fd10801",
        "egoemg_repro/train.py": "10696073b661823a9bf47ddda26b4c3b27324b00c9b62e3841a670409e872b6a",
        "egoemg_repro/vision_literal_runtime.py": "bb430267f819c1a1c46ed961397a81c3b9619d9a48f321939cc78808bc85e4d0",
        "egoemg_repro/visual.py": "b8c390c0291abd67ee58a6f46caddce38296f2013b9f19088c330792b3d76cdb"
      },
      "tensor_inventory": {
        "backbone.0.weight": {
          "dtype": "torch.float32",
          "sha256": "213d57dffee38f02e590f556453243f865a689d38de01daf299cb86545cd29fa",
          "shape": [
            64,
            3,
            7,
            7
          ]
        },
        "backbone.1.bias": {
          "dtype": "torch.float32",
          "sha256": "5cf09c4d1b8c18ba2a7b1c7766dc682339bd61e0d270b64a99efa171dadb8aef",
          "shape": [
            64
          ]
        },
        "backbone.1.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "1c7bf994506d7fce9e3b80f32227f3b25a5b681c0ef8337965963d08631c58a4",
          "shape": []
        },
        "backbone.1.running_mean": {
          "dtype": "torch.float32",
          "sha256": "a58fb1fe496b8e1e0dad5eecf71302acf6076445b9f165fe02e1c092d4fe3f6a",
          "shape": [
            64
          ]
        },
        "backbone.1.running_var": {
          "dtype": "torch.float32",
          "sha256": "ad3717a7c55d411b56b372a786310052a3c563cc9d27f8b77cfbd646e8e0b5ce",
          "shape": [
            64
          ]
        },
        "backbone.1.weight": {
          "dtype": "torch.float32",
          "sha256": "a376cd5dd5b2f95c5f4393a1d3bb11f791736c9271d6f9c5596f7cbf5f63def3",
          "shape": [
            64
          ]
        },
        "backbone.4.0.bn1.bias": {
          "dtype": "torch.float32",
          "sha256": "bb9e9d99420778da95e3acb1a3db8c99170da7f43f964de4cd3d74e38cada817",
          "shape": [
            64
          ]
        },
        "backbone.4.0.bn1.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "8468d9b503180a92d8fcca731b8a00242788f5e7cb15e81d57355e39a8ed343e",
          "shape": []
        },
        "backbone.4.0.bn1.running_mean": {
          "dtype": "torch.float32",
          "sha256": "327b00215d946d37d8ce2bd5acf75608062dd9ce926b617868b3aba22dde8143",
          "shape": [
            64
          ]
        },
        "backbone.4.0.bn1.running_var": {
          "dtype": "torch.float32",
          "sha256": "b3945015fb7cc534e0b43617dbf9f8f00f248112ad10b2dafe85da704135c2b6",
          "shape": [
            64
          ]
        },
        "backbone.4.0.bn1.weight": {
          "dtype": "torch.float32",
          "sha256": "d74d4ca560135641d0ded647a7dd6058ee7507a88315d056336601ecb1b237d6",
          "shape": [
            64
          ]
        },
        "backbone.4.0.bn2.bias": {
          "dtype": "torch.float32",
          "sha256": "53ae11ec4955036ed62378379aa9d2c73cc2666d89aa62095d7d64f7d3f2a718",
          "shape": [
            64
          ]
        },
        "backbone.4.0.bn2.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "7f83a3bc340856fee649324560c1bfc47d6407a38d964afccc70e59f270431e5",
          "shape": []
        },
        "backbone.4.0.bn2.running_mean": {
          "dtype": "torch.float32",
          "sha256": "c63a3a041a6b8563965b6d5972fe20282ade8d8686f8016d94561f64779b2358",
          "shape": [
            64
          ]
        },
        "backbone.4.0.bn2.running_var": {
          "dtype": "torch.float32",
          "sha256": "bcf5b848e8c6607eed43be8beaca41b60c64d541eee4fb618a15004bf870c3ea",
          "shape": [
            64
          ]
        },
        "backbone.4.0.bn2.weight": {
          "dtype": "torch.float32",
          "sha256": "14eb0f47c434f024c2307d673c7655be8df88dd135176267464e5140c474be98",
          "shape": [
            64
          ]
        },
        "backbone.4.0.conv1.weight": {
          "dtype": "torch.float32",
          "sha256": "7db120299002b6c85933b5b9dd64fb76ba5667e21c24b4e41afb6948c3e17393",
          "shape": [
            64,
            64,
            3,
            3
          ]
        },
        "backbone.4.0.conv2.weight": {
          "dtype": "torch.float32",
          "sha256": "ccfefeda1ab21d38719fca4be92c0e735e5d0fdec7904e986fe208bb4850d4c2",
          "shape": [
            64,
            64,
            3,
            3
          ]
        },
        "backbone.4.1.bn1.bias": {
          "dtype": "torch.float32",
          "sha256": "6e673c56b2afcb684a2b5cb7fed873505127401a42bb68191b5ec2ae0aafd568",
          "shape": [
            64
          ]
        },
        "backbone.4.1.bn1.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "52057987cc8bcbf065c3da183010042c45863ad8fbd8970d2bf14c23acf0f7e1",
          "shape": []
        },
        "backbone.4.1.bn1.running_mean": {
          "dtype": "torch.float32",
          "sha256": "20a7d4fc7735cd0188faff20e88b23146484806bed5aa5c8461d7062b82b4ad3",
          "shape": [
            64
          ]
        },
        "backbone.4.1.bn1.running_var": {
          "dtype": "torch.float32",
          "sha256": "512edbdc3d27ebbd75b75ca7139492213def764ee956932277c90770d87d9f79",
          "shape": [
            64
          ]
        },
        "backbone.4.1.bn1.weight": {
          "dtype": "torch.float32",
          "sha256": "d064decfbdbed5b42318e694f1418fdc638fa75cdb5eb774e2b64c97442c9a5b",
          "shape": [
            64
          ]
        },
        "backbone.4.1.bn2.bias": {
          "dtype": "torch.float32",
          "sha256": "547907d6fc4da7a4de8ea2d26a3d21f6e16c07a959e773dd34dfb520cb0278b7",
          "shape": [
            64
          ]
        },
        "backbone.4.1.bn2.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "2dcaeb3be4994379778f0266dda7b938351c95ec7a1c2c4ef2a9dbbec51ab248",
          "shape": []
        },
        "backbone.4.1.bn2.running_mean": {
          "dtype": "torch.float32",
          "sha256": "80b6ac5ecbc0c880e96e03f57bcb8c243c55cb455ff483f4770d4502cba73809",
          "shape": [
            64
          ]
        },
        "backbone.4.1.bn2.running_var": {
          "dtype": "torch.float32",
          "sha256": "37ed2dbfb37b43c467a5bf9f6a148b2b6e778172f067dc7f0964c7e19420c3ce",
          "shape": [
            64
          ]
        },
        "backbone.4.1.bn2.weight": {
          "dtype": "torch.float32",
          "sha256": "eb72b3dd3fc11839679494fa997f175889a06391d882becf9373f00191008498",
          "shape": [
            64
          ]
        },
        "backbone.4.1.conv1.weight": {
          "dtype": "torch.float32",
          "sha256": "2db8f3472b2fa06bd09b6c55f4429cb49560ac59b69ca9317089f4df719d22f1",
          "shape": [
            64,
            64,
            3,
            3
          ]
        },
        "backbone.4.1.conv2.weight": {
          "dtype": "torch.float32",
          "sha256": "d36db7743c048da8ea7ab8347e94e608d3d5c8ca0bd515e2d3df1adaff6e39e9",
          "shape": [
            64,
            64,
            3,
            3
          ]
        },
        "backbone.5.0.bn1.bias": {
          "dtype": "torch.float32",
          "sha256": "cbba26ca2497d36449650dea14c3388e40454f4507726a3ab54d8818366ca634",
          "shape": [
            128
          ]
        },
        "backbone.5.0.bn1.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "4aa8ecfb9013c18c61343a46bb73dfe019bb284d47c57e019f9d4eae4390f901",
          "shape": []
        },
        "backbone.5.0.bn1.running_mean": {
          "dtype": "torch.float32",
          "sha256": "9dd97c1641b72a9693978247447582e0a18f6066662dab12743229a97652bba4",
          "shape": [
            128
          ]
        },
        "backbone.5.0.bn1.running_var": {
          "dtype": "torch.float32",
          "sha256": "9744c0e015d6a2ee55b566e85fd1b90e89243270780d41933fe052c035d45fdc",
          "shape": [
            128
          ]
        },
        "backbone.5.0.bn1.weight": {
          "dtype": "torch.float32",
          "sha256": "26d4d06d040950784031e0c65def6fdabca5b0b46cb3d8ac79bd1b4a7c73a760",
          "shape": [
            128
          ]
        },
        "backbone.5.0.bn2.bias": {
          "dtype": "torch.float32",
          "sha256": "69fb514943bc04003517ab7cc5cf01a9a17c2cfa85db210a9bcb0d09b52010dd",
          "shape": [
            128
          ]
        },
        "backbone.5.0.bn2.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "d1836d3a02835cac75e72d71ea7a9cc5f2d2be5c7502e957e60b3f4f8311f4a0",
          "shape": []
        },
        "backbone.5.0.bn2.running_mean": {
          "dtype": "torch.float32",
          "sha256": "86a84c681d66747b3a1f278c638216f0fd7f431b8b98a05233a55ad5a4f6a702",
          "shape": [
            128
          ]
        },
        "backbone.5.0.bn2.running_var": {
          "dtype": "torch.float32",
          "sha256": "ff145fe3827fcc9d2df6f4e2a82592c12c2e6001516b25d9de49236c79c823b3",
          "shape": [
            128
          ]
        },
        "backbone.5.0.bn2.weight": {
          "dtype": "torch.float32",
          "sha256": "6ce3f8c5697e355ae4192ac4cda522200dca4e4e15325f72ecb6e51a4db59a51",
          "shape": [
            128
          ]
        },
        "backbone.5.0.conv1.weight": {
          "dtype": "torch.float32",
          "sha256": "e83c079ef90003940c59a52650aca03f1f3e327addb5fcd169845063efa11692",
          "shape": [
            128,
            64,
            3,
            3
          ]
        },
        "backbone.5.0.conv2.weight": {
          "dtype": "torch.float32",
          "sha256": "c1679c25f8787015d9708f127f119a2a2c252f4c6a76d33544fd6322ec5c2d43",
          "shape": [
            128,
            128,
            3,
            3
          ]
        },
        "backbone.5.0.downsample.0.weight": {
          "dtype": "torch.float32",
          "sha256": "a3dc1acd00af81df88da6ad3db2a54626bb091e64efaf9f5757cc3b26229a3e0",
          "shape": [
            128,
            64,
            1,
            1
          ]
        },
        "backbone.5.0.downsample.1.bias": {
          "dtype": "torch.float32",
          "sha256": "a037465e2165eb306c4d2eb6119cb6a0186698c0e36b21d7b651cebb907c6132",
          "shape": [
            128
          ]
        },
        "backbone.5.0.downsample.1.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "53f8d321fa83834e1affb8fceaf717101ebd092735b3d424e9cc50507078c76d",
          "shape": []
        },
        "backbone.5.0.downsample.1.running_mean": {
          "dtype": "torch.float32",
          "sha256": "0e2872641bccb61a6fbbfc33b30e4c7430a68df6b3407f29a36be8eae049ea7d",
          "shape": [
            128
          ]
        },
        "backbone.5.0.downsample.1.running_var": {
          "dtype": "torch.float32",
          "sha256": "b1dc46a23262d18bb67055b7138fd01f12194be88ec4f556ed19542ad5046f32",
          "shape": [
            128
          ]
        },
        "backbone.5.0.downsample.1.weight": {
          "dtype": "torch.float32",
          "sha256": "06c89ff14cfddf2699e1a60626038eddbf94c2023b4d4806f28325d7affd7df6",
          "shape": [
            128
          ]
        },
        "backbone.5.1.bn1.bias": {
          "dtype": "torch.float32",
          "sha256": "aa4d9876fb3ae2e90e5b01744f3cdab14248bd7d4d8ab405d20e6071679365bc",
          "shape": [
            128
          ]
        },
        "backbone.5.1.bn1.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "51947480dfe8b580e06e3045f88e036d21c1ef6b961133610dcbfa315a18490e",
          "shape": []
        },
        "backbone.5.1.bn1.running_mean": {
          "dtype": "torch.float32",
          "sha256": "cfd4fff3962a107bc664a07485f95da20c44c0cb6abd42c285c8a91b7ed3e30e",
          "shape": [
            128
          ]
        },
        "backbone.5.1.bn1.running_var": {
          "dtype": "torch.float32",
          "sha256": "374501556362ff7787f27d2577f272de2282dd77b32206ce2b783cb6cc8ca85b",
          "shape": [
            128
          ]
        },
        "backbone.5.1.bn1.weight": {
          "dtype": "torch.float32",
          "sha256": "3d5d8a23cf662f68138f6119d207ea138fc3948ddc6bf918504a2d1fa88b1264",
          "shape": [
            128
          ]
        },
        "backbone.5.1.bn2.bias": {
          "dtype": "torch.float32",
          "sha256": "b24392a2415a8a5dc4f05ac412fa64165e551570e34a270b611f05a37c735822",
          "shape": [
            128
          ]
        },
        "backbone.5.1.bn2.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "f0a94a283594e7967495eebe67bccdddbd0453b2029b34c63d4532bc16730182",
          "shape": []
        },
        "backbone.5.1.bn2.running_mean": {
          "dtype": "torch.float32",
          "sha256": "f57022f4f188f99882b668da1c0b8111fc2961b4ccf7ed4e3186fcac8c5a90e5",
          "shape": [
            128
          ]
        },
        "backbone.5.1.bn2.running_var": {
          "dtype": "torch.float32",
          "sha256": "c4fc0bef3d533b60a88fc6c654bc94e7db96e5fbde04b7394faf5575b86ef77a",
          "shape": [
            128
          ]
        },
        "backbone.5.1.bn2.weight": {
          "dtype": "torch.float32",
          "sha256": "fcfeb4f8c240eeafa74ad0b0a51d01c20229b6cedd52c4049702b0cf83e87fa7",
          "shape": [
            128
          ]
        },
        "backbone.5.1.conv1.weight": {
          "dtype": "torch.float32",
          "sha256": "f709b6105bf828ece64aee25a9bf028876a8e8002accdc31d31088cb15c6cf62",
          "shape": [
            128,
            128,
            3,
            3
          ]
        },
        "backbone.5.1.conv2.weight": {
          "dtype": "torch.float32",
          "sha256": "98da087c626cf79172c69f3eb65478f51a9b7e2110df388bed79a5bf9573aed8",
          "shape": [
            128,
            128,
            3,
            3
          ]
        },
        "backbone.6.0.bn1.bias": {
          "dtype": "torch.float32",
          "sha256": "2eb41c5a79e9f22abed64aa221de242b894681f602bbf1aaa6cb55419407f200",
          "shape": [
            256
          ]
        },
        "backbone.6.0.bn1.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "7f08900e5d527018a41bb6aaeee5da46b596c6f7796c6568d46fed5700214dbd",
          "shape": []
        },
        "backbone.6.0.bn1.running_mean": {
          "dtype": "torch.float32",
          "sha256": "da493ca60fa65d10547692d449f8fa6c9b7c01dd7c6f18a7114859b4f0c14b9d",
          "shape": [
            256
          ]
        },
        "backbone.6.0.bn1.running_var": {
          "dtype": "torch.float32",
          "sha256": "aa5d34e77d8f9de91bc6d8b0d9f46fe55c6648ab5dc87631b0783351eb288a67",
          "shape": [
            256
          ]
        },
        "backbone.6.0.bn1.weight": {
          "dtype": "torch.float32",
          "sha256": "35f94b7361cce5e33026add263d03d7e79ad44af905fd7dc2f2835534195af03",
          "shape": [
            256
          ]
        },
        "backbone.6.0.bn2.bias": {
          "dtype": "torch.float32",
          "sha256": "c01d1cb22ae8fc5595f59f93b3727e68ed275197abfa191a0aea997b450545fa",
          "shape": [
            256
          ]
        },
        "backbone.6.0.bn2.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "88d3d77ebc7349857e9d2fcf735878ea8ea1452ce8e5bf32d06239a32c77c327",
          "shape": []
        },
        "backbone.6.0.bn2.running_mean": {
          "dtype": "torch.float32",
          "sha256": "27d84e94032ce2ce5c31ecb07ad73cc43ff816d6fe12abe62d53bd33fb8b09b1",
          "shape": [
            256
          ]
        },
        "backbone.6.0.bn2.running_var": {
          "dtype": "torch.float32",
          "sha256": "cfb6b8f6fd0d837dbef4f03ae2a2327eae502348ea6f24e9a2cc57b5ea9419b2",
          "shape": [
            256
          ]
        },
        "backbone.6.0.bn2.weight": {
          "dtype": "torch.float32",
          "sha256": "ed7fa05bcba8183608653a5e8106c8eca15df7546c8cb04cda6f631fc77d9d08",
          "shape": [
            256
          ]
        },
        "backbone.6.0.conv1.weight": {
          "dtype": "torch.float32",
          "sha256": "efda9f25ee7d6cc4b8d7385f59b80b31edd9a2c3c87130a35905668490e34e6a",
          "shape": [
            256,
            128,
            3,
            3
          ]
        },
        "backbone.6.0.conv2.weight": {
          "dtype": "torch.float32",
          "sha256": "fa2353a38af33a0df5333c73936292a1ae96186794035a889f52a939089e577e",
          "shape": [
            256,
            256,
            3,
            3
          ]
        },
        "backbone.6.0.downsample.0.weight": {
          "dtype": "torch.float32",
          "sha256": "1272d07e372dc6fe5d6995bfd5007622a4300b6f1130a9ff44128f9f016d0d4f",
          "shape": [
            256,
            128,
            1,
            1
          ]
        },
        "backbone.6.0.downsample.1.bias": {
          "dtype": "torch.float32",
          "sha256": "5eef383b5062433c58d85ed03da6038e02cc0cd1c0588d261aeaee1e13c6cc02",
          "shape": [
            256
          ]
        },
        "backbone.6.0.downsample.1.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "d3d34a4f7b0dff5005e74f015cbde986b00b7d2bb628a625e9e598205c1db43e",
          "shape": []
        },
        "backbone.6.0.downsample.1.running_mean": {
          "dtype": "torch.float32",
          "sha256": "a673000dcf0fd3c61712bad7b083b61e5810624514e55f145a4a3ef6899f8f49",
          "shape": [
            256
          ]
        },
        "backbone.6.0.downsample.1.running_var": {
          "dtype": "torch.float32",
          "sha256": "80ff2baa162c8b34397940eace1fa30670b705eac3d34233288dab38725f57df",
          "shape": [
            256
          ]
        },
        "backbone.6.0.downsample.1.weight": {
          "dtype": "torch.float32",
          "sha256": "82f232462457fe95f29de8dc57e8bbca86c4bf30a2c566d795630e9f6d2ab2ea",
          "shape": [
            256
          ]
        },
        "backbone.6.1.bn1.bias": {
          "dtype": "torch.float32",
          "sha256": "302f1656b4305ea9adf03d63cbabff022a0a0d4e3f551f2002cf4652a2467a8a",
          "shape": [
            256
          ]
        },
        "backbone.6.1.bn1.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "cb8942fc0959fa47bcdb1e75b5d5252881b33857bbbb736230b662c0733d98a4",
          "shape": []
        },
        "backbone.6.1.bn1.running_mean": {
          "dtype": "torch.float32",
          "sha256": "3c840f74bc4879cda9f62a05bcb0b7ffc1558a8c4416ae1656ccc4b85d3cc15c",
          "shape": [
            256
          ]
        },
        "backbone.6.1.bn1.running_var": {
          "dtype": "torch.float32",
          "sha256": "64ecacd4da638440ab99b3e970895d98ed9d2bf4a88fec7612107a86c73ddadc",
          "shape": [
            256
          ]
        },
        "backbone.6.1.bn1.weight": {
          "dtype": "torch.float32",
          "sha256": "011763d7bd99121db61374c09150a4c3fc02508dd8cf820bc969b9bbd554070c",
          "shape": [
            256
          ]
        },
        "backbone.6.1.bn2.bias": {
          "dtype": "torch.float32",
          "sha256": "7b3ebc02cfdf579f22cfdca7e62ac2f2584710df23aa6f83777b1da5dd023ac7",
          "shape": [
            256
          ]
        },
        "backbone.6.1.bn2.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "a8bade65ba936991d3edfa4c2fa02a63033fba4e6c8484b62af8cc156363f67a",
          "shape": []
        },
        "backbone.6.1.bn2.running_mean": {
          "dtype": "torch.float32",
          "sha256": "fd5b2494e0f12c02431d3b153e1dd3e2c208ef1670a68b7a33f11714ce5e032c",
          "shape": [
            256
          ]
        },
        "backbone.6.1.bn2.running_var": {
          "dtype": "torch.float32",
          "sha256": "4c14128ae1730b9284d4e0c031cb210298b91951da13c5a4e921f3c4072f4042",
          "shape": [
            256
          ]
        },
        "backbone.6.1.bn2.weight": {
          "dtype": "torch.float32",
          "sha256": "4dfc71e2acf2ba1819cb8b07d82788b21d2ef556a908183c8abd1ba4936732e9",
          "shape": [
            256
          ]
        },
        "backbone.6.1.conv1.weight": {
          "dtype": "torch.float32",
          "sha256": "a1a556df6ad6ca9d46cbbbe4bd8f213dda77d017e3e1e313b841674fe722e4de",
          "shape": [
            256,
            256,
            3,
            3
          ]
        },
        "backbone.6.1.conv2.weight": {
          "dtype": "torch.float32",
          "sha256": "d492515f46a92739058f89938fe52d69975957e459fcfe56e53e3d7a98e14dfd",
          "shape": [
            256,
            256,
            3,
            3
          ]
        },
        "backbone.7.0.bn1.bias": {
          "dtype": "torch.float32",
          "sha256": "17f490f980432c68b02f156e01f592563bdbdc98638b5f09f97dbc2d38888539",
          "shape": [
            512
          ]
        },
        "backbone.7.0.bn1.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "9687f952409405079931d9e2e05d4dff69fbaa3b4f83c15d70f5794a2a804f10",
          "shape": []
        },
        "backbone.7.0.bn1.running_mean": {
          "dtype": "torch.float32",
          "sha256": "5f2df200455959b74aa498a9a3fe878fb0133ce774cc925c4069d01cd544e70c",
          "shape": [
            512
          ]
        },
        "backbone.7.0.bn1.running_var": {
          "dtype": "torch.float32",
          "sha256": "6ad0445bb7fd0326d54ac42f7825e2cf1fcb6b064f016813328aadcdb6ee4681",
          "shape": [
            512
          ]
        },
        "backbone.7.0.bn1.weight": {
          "dtype": "torch.float32",
          "sha256": "802511d6c9b40da98467cbd165a5f2760e80f1ddab5be1f8fd4ca5fd71173a23",
          "shape": [
            512
          ]
        },
        "backbone.7.0.bn2.bias": {
          "dtype": "torch.float32",
          "sha256": "5dab38917de6fba127c7827dc4beac8621284653143c04ba089a61d1eaef44c0",
          "shape": [
            512
          ]
        },
        "backbone.7.0.bn2.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "0883bc45e650d3b125176aef6770b4ed319f188c08a290ad77d08f49b1f2a960",
          "shape": []
        },
        "backbone.7.0.bn2.running_mean": {
          "dtype": "torch.float32",
          "sha256": "c2125e3c169f4080aa83eab54dc786f736ba9d6aeb198079b70e4e817646c923",
          "shape": [
            512
          ]
        },
        "backbone.7.0.bn2.running_var": {
          "dtype": "torch.float32",
          "sha256": "69c2934f37bbf72681b9df975ea55801724f5c10eb2141a92abf3cb356622d03",
          "shape": [
            512
          ]
        },
        "backbone.7.0.bn2.weight": {
          "dtype": "torch.float32",
          "sha256": "ce349a4923d5a7162a7398ae88ab325d18ec52051d70d9feab31032f8f6ce1d0",
          "shape": [
            512
          ]
        },
        "backbone.7.0.conv1.weight": {
          "dtype": "torch.float32",
          "sha256": "3893decedd5c7f61cf10438ea5d943312d502000b45513c285b7c95972774dad",
          "shape": [
            512,
            256,
            3,
            3
          ]
        },
        "backbone.7.0.conv2.weight": {
          "dtype": "torch.float32",
          "sha256": "03c12f381543f70ae9c13d1c023bbe42a57875bcbba6625a359ddd4222a517de",
          "shape": [
            512,
            512,
            3,
            3
          ]
        },
        "backbone.7.0.downsample.0.weight": {
          "dtype": "torch.float32",
          "sha256": "0da89e8c01b157e256d49e4c3042aa4d312791a55a8ebc05078b2e0933b07864",
          "shape": [
            512,
            256,
            1,
            1
          ]
        },
        "backbone.7.0.downsample.1.bias": {
          "dtype": "torch.float32",
          "sha256": "b9f46ffbc511bf1a327e52be175378128d221a125780c8dc400fc57790a7c133",
          "shape": [
            512
          ]
        },
        "backbone.7.0.downsample.1.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "6a552e8ec6de1eb04998e9b299b161585287e0e2ea6d73df729807129391b564",
          "shape": []
        },
        "backbone.7.0.downsample.1.running_mean": {
          "dtype": "torch.float32",
          "sha256": "5aca79c2487584657bfab200b796cfd068007c4e2a1249816cf990e717cd1b11",
          "shape": [
            512
          ]
        },
        "backbone.7.0.downsample.1.running_var": {
          "dtype": "torch.float32",
          "sha256": "2bd5a613c29921f7c77594a053cb9029004cf5abb82f8d2d54604c8941ac7fee",
          "shape": [
            512
          ]
        },
        "backbone.7.0.downsample.1.weight": {
          "dtype": "torch.float32",
          "sha256": "97630cb1015b4d609c21e54a6633cf397fe777538de928e26609a3c1f9317898",
          "shape": [
            512
          ]
        },
        "backbone.7.1.bn1.bias": {
          "dtype": "torch.float32",
          "sha256": "ed7c44b78b3a363bdd08658e94debb691a14011376c1346744c95ad8a4bfd1d8",
          "shape": [
            512
          ]
        },
        "backbone.7.1.bn1.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "954cab30d6a794175ef4d8380ce73fc07aa30d03ce3f18097e0ca82aedd32d2a",
          "shape": []
        },
        "backbone.7.1.bn1.running_mean": {
          "dtype": "torch.float32",
          "sha256": "d3e4a5e85a331b98a9f82021109619b7653f41f4f565e142d5e63101daa0cc94",
          "shape": [
            512
          ]
        },
        "backbone.7.1.bn1.running_var": {
          "dtype": "torch.float32",
          "sha256": "c1b176906d1fd06b7dbddc14a200af997767f985af30c024bd7890b674500488",
          "shape": [
            512
          ]
        },
        "backbone.7.1.bn1.weight": {
          "dtype": "torch.float32",
          "sha256": "d9e542fd33270b602bb7df9654e42e87c20a1c1306d830506deca510c6acfc46",
          "shape": [
            512
          ]
        },
        "backbone.7.1.bn2.bias": {
          "dtype": "torch.float32",
          "sha256": "309afa4a5e4f9e536522ab0187d43a9367bebfb80f0309c3ab259baff4ab2319",
          "shape": [
            512
          ]
        },
        "backbone.7.1.bn2.num_batches_tracked": {
          "dtype": "torch.int64",
          "sha256": "840b9d8b3f7824169b19fb2cce1dfd68e161b887b26366e820f5374a02924c81",
          "shape": []
        },
        "backbone.7.1.bn2.running_mean": {
          "dtype": "torch.float32",
          "sha256": "e34e631b7fd2ee5c04bf87d16cd65d3ed0028415d34d0ffc031aea371699affd",
          "shape": [
            512
          ]
        },
        "backbone.7.1.bn2.running_var": {
          "dtype": "torch.float32",
          "sha256": "9d949fbf558f755fc68105dfb70f1d639d1ce745cc80de858f5ee3214029cb0e",
          "shape": [
            512
          ]
        },
        "backbone.7.1.bn2.weight": {
          "dtype": "torch.float32",
          "sha256": "0bd80284ee3f52ab07739638a91378d5f1c2d2a77a8b4d73e60e9a4397a6defd",
          "shape": [
            512
          ]
        },
        "backbone.7.1.conv1.weight": {
          "dtype": "torch.float32",
          "sha256": "d868de32e44fe81f8c2431f77d5a3e15445f37287f736c8a0db6f2300f0b2f4d",
          "shape": [
            512,
            512,
            3,
            3
          ]
        },
        "backbone.7.1.conv2.weight": {
          "dtype": "torch.float32",
          "sha256": "a4a262c16dfe57b10c3a3bf80e1938f6380bc735ae4267455744f22df9f3f759",
          "shape": [
            512,
            512,
            3,
            3
          ]
        },
        "head.0.bias": {
          "dtype": "torch.float32",
          "sha256": "83b8712393eb60580f9e2adeb7f43fd3b28e3b43060126fa0e3339e72187b85a",
          "shape": [
            512
          ]
        },
        "head.0.weight": {
          "dtype": "torch.float32",
          "sha256": "748c4228f31bae47d4f186fc0b87d08dad0632e09ccd0845ac640b1b0bb7227b",
          "shape": [
            512,
            512
          ]
        },
        "head.3.bias": {
          "dtype": "torch.float32",
          "sha256": "8f74b520c48e94d019f6e7ee7145817b8f0bee4a4a73e7b43b771d35522e253c",
          "shape": [
            22
          ]
        },
        "head.3.weight": {
          "dtype": "torch.float32",
          "sha256": "eb441fe340079f7fff4e5328b78e77762066e6e6628ce77f77f470a151ebe48a",
          "shape": [
            22,
            512
          ]
        }
      }
    }
  },
  "native_metrics": {
    "test_derivatives/acc": 0.0,
    "test_derivatives/jerk": 0.0,
    "test_derivatives/vel": 0.0,
    "test_landmark/all": 8.619263648986816,
    "test_landmark/fingertip": 13.719047546386719,
    "test_loss": 0.38232421875,
    "test_mae": 0.10794329643249512,
    "test_mae_per_finger/index": 0.09812993556261063,
    "test_mae_per_finger/middle": 0.10212015360593796,
    "test_mae_per_finger/pinky": 0.1418478935956955,
    "test_mae_per_finger/ring": 0.10290312021970749,
    "test_mae_per_finger/thumb": 0.09820083528757095,
    "test_mae_per_pd/distal": 0.0890001505613327,
    "test_mae_per_pd/mid": 0.13191048800945282,
    "test_mae_per_pd/proximal": 0.10682545602321625,
    "test_mae_wrist/deviation": 0.10635223239660263,
    "test_mae_wrist/flexion": 0.09559246897697449
  },
  "native_test_sha256": "3c1bb0e2c566352477c821ad36774afe6378ca68d9790ed315cb3fe58a8bdc5b",
  "paper_numerical_equivalence_established": false,
  "plot_runtime": {
    "backend": "Agg",
    "font_sha256": "3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22",
    "freetype_version": "2.6.1",
    "matplotlib_version": "3.9.4",
    "svg_hashsalt": "egoemg-original-rn18-fusion-v1"
  },
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
  "prediction_sha256": "3b07d497182b2c7dfca5b08de31e68cf276ddb294d2945c7d1da02d8a3321ca3",
  "saved_fit_validation": {
    "completed_epoch": 145,
    "evaluation_native_mae_rad": 0.10630998760461807,
    "evaluation_pose_sha256": "f3c2f42b11a0347e056d1b4ab930cf192d3ec9162a5e7369bb051f306ece3eb8",
    "fit_native_mae_rad": 0.10630998760461807,
    "fit_pose_sha256": "f3c2f42b11a0347e056d1b4ab930cf192d3ec9162a5e7369bb051f306ece3eb8",
    "native_mae_difference_rad": 0.0,
    "pose_exact_match": true,
    "view_policy": "exact typed selected-fit validation pose and fixed-tolerance native MAE agreement; standalone independent replay also exact"
  },
  "scientific_eligible": true,
  "scope": "available legacy cohort; single rank; original full200/10worker loaders; original fusion semantics; independent train-only validation",
  "sha256": "ab1fb5a8740c479ab448d0543c69ccd70582dd370d74cb1db0a6d5755bd5de81",
  "sources_sha256": {
    "egoemg_repro/classic_emg_report.py": "bdddaf770519151c24cd0f8fdbd9f40e4453e04fa1a28e15088ad825af325868",
    "egoemg_repro/emg2pose_memory_process.py": "8dd958fdbe9b2a14258a1b1b225d3806dc86f7f903cbbf7775e298d1041528db",
    "egoemg_repro/emgformer_trajectory_eval.py": "0a85fc8df618113cba2181e62a8229e90542e91e5d4804901d430e08494a4a5c",
    "egoemg_repro/metrics.py": "12d6eb170997ae78ddbf5602d1252d05a778f1357f17b28aa3b96d06a618798e",
    "egoemg_repro/rn18_fusion_eval.py": "4b45bfe0d1ba13a5bb7e69250a34271598731d3350a111b63b3410bc5b200df1",
    "egoemg_repro/rn18_fusion_train.py": "a5cf509288f95d5cafdec046cc343fc5d051481695680f7aab9d095f1e6d86b4",
    "egoemg_repro/vision_literal_eval.py": "63553900476855dc0cf9ecc2f97f841e33225470488066de39af23465565c581"
  },
  "test": {
    "batch_sizes": [
      200,
      200,
      200,
      200,
      200,
      200,
      200,
      200,
      200,
      200,
      200,
      200,
      200,
      192
    ],
    "consumed": {
      "all": {
        "by_component": {
          "fingers": {
            "absolute_error_sum_rad": 5886.619335478128,
            "mae_deg": 6.227445412167689,
            "valid_angle_count": 54160
          },
          "wrist": {
            "absolute_error_sum_rad": 548.787916780333,
            "mae_deg": 5.805618810799434,
            "valid_angle_count": 5416
          }
        },
        "by_finger": {
          "index": {
            "absolute_error_sum_rad": 1063.3266824146194,
            "mae_deg": 5.624458192947318,
            "valid_angle_count": 10832
          },
          "middle": {
            "absolute_error_sum_rad": 1105.1263062367361,
            "mae_deg": 5.845556977127689,
            "valid_angle_count": 10832
          },
          "pinky": {
            "absolute_error_sum_rad": 1537.96813735889,
            "mae_deg": 8.135070466789225,
            "valid_angle_count": 10832
          },
          "ring": {
            "absolute_error_sum_rad": 1115.3588544936065,
            "mae_deg": 5.899681961321062,
            "valid_angle_count": 10832
          },
          "thumb": {
            "absolute_error_sum_rad": 1064.839354974276,
            "mae_deg": 5.6324594626531495,
            "valid_angle_count": 10832
          }
        },
        "by_hand": {
          "left": {
            "absolute_error_sum_rad": 3337.5490544303757,
            "mae_deg": 6.424359159334113,
            "valid_angle_count": 29766
          },
          "right": {
            "absolute_error_sum_rad": 3097.8581978280854,
            "mae_deg": 5.954183168921588,
            "valid_angle_count": 29810
          }
        },
        "by_hand_split": {
          "left": {
            "both": {
              "absolute_error_sum_rad": 276.5194798059383,
              "mae_deg": 7.201545066378754,
              "valid_angle_count": 2200
            },
            "gesture": {
              "absolute_error_sum_rad": 1638.5059665560839,
              "mae_deg": 5.877753355288456,
              "valid_angle_count": 15972
            },
            "user": {
              "absolute_error_sum_rad": 1422.5236080683535,
              "mae_deg": 7.02989468691036,
              "valid_angle_count": 11594
            }
          },
          "right": {
            "both": {
              "absolute_error_sum_rad": 249.58466500695795,
              "mae_deg": 6.435710142252555,
              "valid_angle_count": 2222
            },
            "gesture": {
              "absolute_error_sum_rad": 1567.5258714579613,
              "mae_deg": 5.6308850603202245,
              "valid_angle_count": 15950
            },
            "user": {
              "absolute_error_sum_rad": 1280.7476613631661,
              "mae_deg": 6.3053304362742555,
              "valid_angle_count": 11638
            }
          }
        },
        "by_joint": [
          {
            "absolute_error_sum_rad": 206.45633886333962,
            "index": 0,
            "mae_deg": 4.368196776437266,
            "name": "THUMB_CMC_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 252.52317913815205,
            "index": 1,
            "mae_deg": 5.34287754573196,
            "name": "THUMB_CMC_AA",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 288.57564913039096,
            "index": 2,
            "mae_deg": 6.105674581026398,
            "name": "THUMB_MCP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 317.2841878423933,
            "index": 3,
            "mae_deg": 6.713088947416972,
            "name": "THUMB_IP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 231.65402994747274,
            "index": 4,
            "mae_deg": 4.901328738252351,
            "name": "INDEX_MCP_AA",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 266.88148364319204,
            "index": 5,
            "mae_deg": 5.646670104484718,
            "name": "INDEX_MCP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 353.9158042225754,
            "index": 6,
            "mae_deg": 7.488139543918718,
            "name": "INDEX_PIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 210.87536460137926,
            "index": 7,
            "mae_deg": 4.461694385133483,
            "name": "INDEX_DIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 223.55420703861455,
            "index": 8,
            "mae_deg": 4.729952937853182,
            "name": "MIDDLE_MCP_AA",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 301.65601108112605,
            "index": 9,
            "mae_deg": 6.382428471085714,
            "name": "MIDDLE_MCP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 342.6658165838453,
            "index": 10,
            "mae_deg": 7.250112656446938,
            "name": "MIDDLE_PIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 237.25027153315023,
            "index": 11,
            "mae_deg": 5.019733843124921,
            "name": "MIDDLE_DIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 199.14583071759262,
            "index": 12,
            "mae_deg": 4.213521273170164,
            "name": "RING_MCP_AA",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 342.22609881625976,
            "index": 13,
            "mae_deg": 7.2408091216391215,
            "name": "RING_MCP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 383.4820558654028,
            "index": 14,
            "mae_deg": 8.113701373739895,
            "name": "RING_PIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 190.5048690943513,
            "index": 15,
            "mae_deg": 4.030696076735068,
            "name": "RING_DIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 358.62375793778483,
            "index": 16,
            "mae_deg": 7.587750281741627,
            "name": "PINKY_MCP_AA",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 512.881078623017,
            "index": 17,
            "mae_deg": 10.851521860124157,
            "name": "PINKY_MCP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 417.57535791536793,
            "index": 18,
            "mae_deg": 8.835046394835803,
            "name": "PINKY_PIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 248.88794288272038,
            "index": 19,
            "mae_deg": 5.265963330455309,
            "name": "PINKY_DIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 259.4772594003007,
            "index": 20,
            "mae_deg": 5.490011758958086,
            "name": "wrist_pitch",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 289.31065738003235,
            "index": 21,
            "mae_deg": 6.121225862640782,
            "name": "wrist_yaw",
            "valid_angle_count": 2708
          }
        ],
        "by_pd_group": {
          "distal": {
            "absolute_error_sum_rad": 1204.8026359539945,
            "mae_deg": 5.09823531657315,
            "valid_angle_count": 13540
          },
          "mid": {
            "absolute_error_sum_rad": 1786.2146837175824,
            "mae_deg": 7.558534909993551,
            "valid_angle_count": 13540
          },
          "proximal": {
            "absolute_error_sum_rad": 2895.602015806551,
            "mae_deg": 6.126505711052027,
            "valid_angle_count": 27080
          }
        },
        "by_split": {
          "both": {
            "absolute_error_sum_rad": 526.1041448128963,
            "mae_deg": 6.816722542315341,
            "valid_angle_count": 4422
          },
          "gesture": {
            "absolute_error_sum_rad": 3206.031838014045,
            "mae_deg": 5.754404276134791,
            "valid_angle_count": 31922
          },
          "user": {
            "absolute_error_sum_rad": 2703.2712694315196,
            "mae_deg": 6.666926421203448,
            "valid_angle_count": 23232
          }
        },
        "by_split_user_summary": {
          "both": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 6.738172038191623,
            "std_mae_deg": 0.4941624571237397,
            "user_count": 5
          },
          "gesture": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 5.819429587851573,
            "std_mae_deg": 1.4249036883218045,
            "user_count": 36
          },
          "user": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 6.669391059000217,
            "std_mae_deg": 0.4387321528163526,
            "user_count": 5
          }
        },
        "by_user": {
          "user_000": {
            "absolute_error_sum_rad": 83.56809724539198,
            "mae_deg": 5.727391476195935,
            "valid_angle_count": 836
          },
          "user_001": {
            "absolute_error_sum_rad": 99.12497071968392,
            "mae_deg": 5.1631295150870455,
            "valid_angle_count": 1100
          },
          "user_002": {
            "absolute_error_sum_rad": 164.41651148995152,
            "mae_deg": 9.308668172567618,
            "valid_angle_count": 1012
          },
          "user_003": {
            "absolute_error_sum_rad": 95.19447868224233,
            "mae_deg": 6.524212752919345,
            "valid_angle_count": 836
          },
          "user_004": {
            "absolute_error_sum_rad": 108.32932391818758,
            "mae_deg": 6.561113169151957,
            "valid_angle_count": 946
          },
          "user_005": {
            "absolute_error_sum_rad": 97.84194498395664,
            "mae_deg": 4.996373000830579,
            "valid_angle_count": 1122
          },
          "user_006": {
            "absolute_error_sum_rad": 94.61737235577311,
            "mae_deg": 5.356893383995639,
            "valid_angle_count": 1012
          },
          "user_007": {
            "absolute_error_sum_rad": 111.67983295151498,
            "mae_deg": 5.288250483345346,
            "valid_angle_count": 1210
          },
          "user_008": {
            "absolute_error_sum_rad": 130.53926660535217,
            "mae_deg": 6.295748347828054,
            "valid_angle_count": 1188
          },
          "user_009": {
            "absolute_error_sum_rad": 76.65201980667189,
            "mae_deg": 4.990724120540332,
            "valid_angle_count": 880
          },
          "user_010": {
            "absolute_error_sum_rad": 127.34546829434112,
            "mae_deg": 5.922368403719792,
            "valid_angle_count": 1232
          },
          "user_011": {
            "absolute_error_sum_rad": 82.46984899601375,
            "mae_deg": 5.652122349943538,
            "valid_angle_count": 836
          },
          "user_012": {
            "absolute_error_sum_rad": 942.6258373313613,
            "mae_deg": 7.054399443452492,
            "valid_angle_count": 7656
          },
          "user_013": {
            "absolute_error_sum_rad": 68.74525755061768,
            "mae_deg": 7.161478398527724,
            "valid_angle_count": 550
          },
          "user_014": {
            "absolute_error_sum_rad": 130.16067571028543,
            "mae_deg": 6.2774893743857945,
            "valid_angle_count": 1188
          },
          "user_015": {
            "absolute_error_sum_rad": 102.79834115653648,
            "mae_deg": 4.95783761718479,
            "valid_angle_count": 1188
          },
          "user_016": {
            "absolute_error_sum_rad": 12.977623669430614,
            "mae_deg": 5.633053517935264,
            "valid_angle_count": 132
          },
          "user_017": {
            "absolute_error_sum_rad": 38.05198563449085,
            "mae_deg": 4.955041315792644,
            "valid_angle_count": 440
          },
          "user_018": {
            "absolute_error_sum_rad": 117.0042771002627,
            "mae_deg": 4.91484696687994,
            "valid_angle_count": 1364
          },
          "user_019": {
            "absolute_error_sum_rad": 114.79638838203391,
            "mae_deg": 5.435825254243951,
            "valid_angle_count": 1210
          },
          "user_020": {
            "absolute_error_sum_rad": 140.54557588358875,
            "mae_deg": 5.719224664321933,
            "valid_angle_count": 1408
          },
          "user_021": {
            "absolute_error_sum_rad": 65.5961377483618,
            "mae_deg": 8.541776921227077,
            "valid_angle_count": 440
          },
          "user_022": {
            "absolute_error_sum_rad": 100.03357705405506,
            "mae_deg": 4.65219299902116,
            "valid_angle_count": 1232
          },
          "user_023": {
            "absolute_error_sum_rad": 85.35190368100302,
            "mae_deg": 4.832316061589514,
            "valid_angle_count": 1012
          },
          "user_024": {
            "absolute_error_sum_rad": 659.6776463734277,
            "mae_deg": 6.817594692698594,
            "valid_angle_count": 5544
          },
          "user_025": {
            "absolute_error_sum_rad": 67.34700430813245,
            "mae_deg": 6.264121931339874,
            "valid_angle_count": 616
          },
          "user_026": {
            "absolute_error_sum_rad": 144.59119399782503,
            "mae_deg": 6.724403547753811,
            "valid_angle_count": 1232
          },
          "user_027": {
            "absolute_error_sum_rad": 113.72656305972487,
            "mae_deg": 4.936403092311097,
            "valid_angle_count": 1320
          },
          "user_028": {
            "absolute_error_sum_rad": 60.478804320213385,
            "mae_deg": 4.632593900462295,
            "valid_angle_count": 748
          },
          "user_029": {
            "absolute_error_sum_rad": 52.60845932923257,
            "mae_deg": 4.2815947247878805,
            "valid_angle_count": 704
          },
          "user_030": {
            "absolute_error_sum_rad": 388.44859625879326,
            "mae_deg": 6.322859410059702,
            "valid_angle_count": 3520
          },
          "user_031": {
            "absolute_error_sum_rad": 37.043130449950695,
            "mae_deg": 7.421031589981542,
            "valid_angle_count": 286
          },
          "user_032": {
            "absolute_error_sum_rad": 31.564522486180067,
            "mae_deg": 4.566954345464399,
            "valid_angle_count": 396
          },
          "user_033": {
            "absolute_error_sum_rad": 54.365869011497125,
            "mae_deg": 5.056712408960802,
            "valid_angle_count": 616
          },
          "user_034": {
            "absolute_error_sum_rad": 56.07862250925973,
            "mae_deg": 4.868285440436243,
            "valid_angle_count": 660
          },
          "user_035": {
            "absolute_error_sum_rad": 59.41968279890716,
            "mae_deg": 5.158328855126478,
            "valid_angle_count": 660
          },
          "user_036": {
            "absolute_error_sum_rad": 71.21993872424355,
            "mae_deg": 4.755946277481925,
            "valid_angle_count": 858
          },
          "user_037": {
            "absolute_error_sum_rad": 623.6433891071792,
            "mae_deg": 5.863494275857622,
            "valid_angle_count": 6094
          },
          "user_038": {
            "absolute_error_sum_rad": 669.6664410493613,
            "mae_deg": 6.976192864304499,
            "valid_angle_count": 5500
          },
          "user_039": {
            "absolute_error_sum_rad": 155.0606715234244,
            "mae_deg": 11.217578341864403,
            "valid_angle_count": 792
          }
        },
        "invalid_angle_count": 1848,
        "mode": "center",
        "overall": {
          "absolute_error_sum_rad": 6435.407252258461,
          "mae_deg": 6.1890975393160295,
          "valid_angle_count": 59576
        },
        "user_summary": {
          "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
          "mean_mae_deg": 5.944664335239466,
          "std_mae_deg": 1.3732357761195295,
          "user_count": 40
        }
      },
      "definition": "all requested labeled centers; vision_valid is the identical crop-present subset",
      "missing_by_hand": {
        "left": 53,
        "right": 51
      },
      "missing_by_split": {
        "both": 7,
        "gesture": 73,
        "user": 24
      },
      "samples": {
        "all": 2792,
        "missing_crop": 104,
        "vision_valid": 2688
      },
      "vision_valid": {
        "by_component": {
          "fingers": {
            "absolute_error_sum_rad": 5736.168137136454,
            "mae_deg": 6.113434241728766,
            "valid_angle_count": 53760
          },
          "wrist": {
            "absolute_error_sum_rad": 538.8793001844315,
            "mae_deg": 5.743212345150909,
            "valid_angle_count": 5376
          }
        },
        "by_hand": {
          "left": {
            "absolute_error_sum_rad": 3260.4284020133055,
            "mae_deg": 6.322642213495758,
            "valid_angle_count": 29546
          },
          "right": {
            "absolute_error_sum_rad": 3014.61903530758,
            "mae_deg": 5.837274334671309,
            "valid_angle_count": 29590
          }
        },
        "by_hand_split": {
          "left": {
            "both": {
              "absolute_error_sum_rad": 276.5194798059383,
              "mae_deg": 7.201545066378754,
              "valid_angle_count": 2200
            },
            "gesture": {
              "absolute_error_sum_rad": 1570.4659475837252,
              "mae_deg": 5.696446610886365,
              "valid_angle_count": 15796
            },
            "user": {
              "absolute_error_sum_rad": 1413.442974623642,
              "mae_deg": 7.011629179943844,
              "valid_angle_count": 11550
            }
          },
          "right": {
            "both": {
              "absolute_error_sum_rad": 249.58466500695795,
              "mae_deg": 6.435710142252555,
              "valid_angle_count": 2222
            },
            "gesture": {
              "absolute_error_sum_rad": 1493.4576207021819,
              "mae_deg": 5.424674689228144,
              "valid_angle_count": 15774
            },
            "user": {
              "absolute_error_sum_rad": 1271.57674959844,
              "mae_deg": 6.28393833698069,
              "valid_angle_count": 11594
            }
          }
        },
        "by_split": {
          "both": {
            "absolute_error_sum_rad": 526.1041448128963,
            "mae_deg": 6.816722542315341,
            "valid_angle_count": 4422
          },
          "gesture": {
            "absolute_error_sum_rad": 3063.923568285907,
            "mae_deg": 5.560655344106612,
            "valid_angle_count": 31570
          },
          "user": {
            "absolute_error_sum_rad": 2685.019724222082,
            "mae_deg": 6.647092037128652,
            "valid_angle_count": 23144
          }
        },
        "by_user": {
          "user_000": {
            "absolute_error_sum_rad": 83.56809724539198,
            "mae_deg": 5.727391476195935,
            "valid_angle_count": 836
          },
          "user_001": {
            "absolute_error_sum_rad": 99.12497071968392,
            "mae_deg": 5.1631295150870455,
            "valid_angle_count": 1100
          },
          "user_002": {
            "absolute_error_sum_rad": 164.41651148995152,
            "mae_deg": 9.308668172567618,
            "valid_angle_count": 1012
          },
          "user_003": {
            "absolute_error_sum_rad": 62.767252259887755,
            "mae_deg": 5.10837875869964,
            "valid_angle_count": 704
          },
          "user_004": {
            "absolute_error_sum_rad": 108.32932391818758,
            "mae_deg": 6.561113169151957,
            "valid_angle_count": 946
          },
          "user_005": {
            "absolute_error_sum_rad": 87.14001729225856,
            "mae_deg": 4.631498346515239,
            "valid_angle_count": 1078
          },
          "user_006": {
            "absolute_error_sum_rad": 94.61737235577311,
            "mae_deg": 5.356893383995639,
            "valid_angle_count": 1012
          },
          "user_007": {
            "absolute_error_sum_rad": 111.67983295151498,
            "mae_deg": 5.288250483345346,
            "valid_angle_count": 1210
          },
          "user_008": {
            "absolute_error_sum_rad": 130.53926660535217,
            "mae_deg": 6.295748347828054,
            "valid_angle_count": 1188
          },
          "user_009": {
            "absolute_error_sum_rad": 76.65201980667189,
            "mae_deg": 4.990724120540332,
            "valid_angle_count": 880
          },
          "user_010": {
            "absolute_error_sum_rad": 127.34546829434112,
            "mae_deg": 5.922368403719792,
            "valid_angle_count": 1232
          },
          "user_011": {
            "absolute_error_sum_rad": 82.46984899601375,
            "mae_deg": 5.652122349943538,
            "valid_angle_count": 836
          },
          "user_012": {
            "absolute_error_sum_rad": 942.6258373313613,
            "mae_deg": 7.054399443452492,
            "valid_angle_count": 7656
          },
          "user_013": {
            "absolute_error_sum_rad": 68.74525755061768,
            "mae_deg": 7.161478398527724,
            "valid_angle_count": 550
          },
          "user_014": {
            "absolute_error_sum_rad": 130.16067571028543,
            "mae_deg": 6.2774893743857945,
            "valid_angle_count": 1188
          },
          "user_015": {
            "absolute_error_sum_rad": 102.79834115653648,
            "mae_deg": 4.95783761718479,
            "valid_angle_count": 1188
          },
          "user_016": {
            "absolute_error_sum_rad": 12.977623669430614,
            "mae_deg": 5.633053517935264,
            "valid_angle_count": 132
          },
          "user_017": {
            "absolute_error_sum_rad": 38.05198563449085,
            "mae_deg": 4.955041315792644,
            "valid_angle_count": 440
          },
          "user_018": {
            "absolute_error_sum_rad": 117.0042771002627,
            "mae_deg": 4.91484696687994,
            "valid_angle_count": 1364
          },
          "user_019": {
            "absolute_error_sum_rad": 114.79638838203391,
            "mae_deg": 5.435825254243951,
            "valid_angle_count": 1210
          },
          "user_020": {
            "absolute_error_sum_rad": 140.54557588358875,
            "mae_deg": 5.719224664321933,
            "valid_angle_count": 1408
          },
          "user_021": {
            "absolute_error_sum_rad": 65.5961377483618,
            "mae_deg": 8.541776921227077,
            "valid_angle_count": 440
          },
          "user_022": {
            "absolute_error_sum_rad": 100.03357705405506,
            "mae_deg": 4.65219299902116,
            "valid_angle_count": 1232
          },
          "user_023": {
            "absolute_error_sum_rad": 85.35190368100302,
            "mae_deg": 4.832316061589514,
            "valid_angle_count": 1012
          },
          "user_024": {
            "absolute_error_sum_rad": 659.6776463734277,
            "mae_deg": 6.817594692698594,
            "valid_angle_count": 5544
          },
          "user_025": {
            "absolute_error_sum_rad": 67.34700430813245,
            "mae_deg": 6.264121931339874,
            "valid_angle_count": 616
          },
          "user_026": {
            "absolute_error_sum_rad": 144.59119399782503,
            "mae_deg": 6.724403547753811,
            "valid_angle_count": 1232
          },
          "user_027": {
            "absolute_error_sum_rad": 113.72656305972487,
            "mae_deg": 4.936403092311097,
            "valid_angle_count": 1320
          },
          "user_028": {
            "absolute_error_sum_rad": 60.478804320213385,
            "mae_deg": 4.632593900462295,
            "valid_angle_count": 748
          },
          "user_029": {
            "absolute_error_sum_rad": 52.60845932923257,
            "mae_deg": 4.2815947247878805,
            "valid_angle_count": 704
          },
          "user_030": {
            "absolute_error_sum_rad": 370.19705104935565,
            "mae_deg": 6.180282229987518,
            "valid_angle_count": 3432
          },
          "user_031": {
            "absolute_error_sum_rad": 21.280478580854833,
            "mae_deg": 5.038353754964998,
            "valid_angle_count": 242
          },
          "user_032": {
            "absolute_error_sum_rad": 31.564522486180067,
            "mae_deg": 4.566954345464399,
            "valid_angle_count": 396
          },
          "user_033": {
            "absolute_error_sum_rad": 54.365869011497125,
            "mae_deg": 5.056712408960802,
            "valid_angle_count": 616
          },
          "user_034": {
            "absolute_error_sum_rad": 56.07862250925973,
            "mae_deg": 4.868285440436243,
            "valid_angle_count": 660
          },
          "user_035": {
            "absolute_error_sum_rad": 59.41968279890716,
            "mae_deg": 5.158328855126478,
            "valid_angle_count": 660
          },
          "user_036": {
            "absolute_error_sum_rad": 71.21993872424355,
            "mae_deg": 4.755946277481925,
            "valid_angle_count": 858
          },
          "user_037": {
            "absolute_error_sum_rad": 623.6433891071792,
            "mae_deg": 5.863494275857622,
            "valid_angle_count": 6094
          },
          "user_038": {
            "absolute_error_sum_rad": 669.6664410493613,
            "mae_deg": 6.976192864304499,
            "valid_angle_count": 5500
          },
          "user_039": {
            "absolute_error_sum_rad": 71.8442077784348,
            "mae_deg": 6.236924072977688,
            "valid_angle_count": 660
          }
        },
        "invalid_angle_count": 0,
        "mode": "center",
        "overall": {
          "absolute_error_sum_rad": 6275.047437320885,
          "mae_deg": 6.079777705676233,
          "valid_angle_count": 59136
        },
        "user_summary": {
          "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
          "mean_mae_deg": 5.712498886926704,
          "std_mae_deg": 1.0641861940103032,
          "user_count": 40
        }
      }
    },
    "consumed_samples": 2792,
    "duplicate_consumptions": 0,
    "format": "original_rn18_fusion_center_statistics_v1",
    "native_batch_weighted_mae_rad": 0.10794328898191452,
    "role": "test",
    "unique_first_repeat": {
      "all": {
        "by_component": {
          "fingers": {
            "absolute_error_sum_rad": 5886.619335478128,
            "mae_deg": 6.227445412167689,
            "valid_angle_count": 54160
          },
          "wrist": {
            "absolute_error_sum_rad": 548.787916780333,
            "mae_deg": 5.805618810799434,
            "valid_angle_count": 5416
          }
        },
        "by_finger": {
          "index": {
            "absolute_error_sum_rad": 1063.3266824146194,
            "mae_deg": 5.624458192947318,
            "valid_angle_count": 10832
          },
          "middle": {
            "absolute_error_sum_rad": 1105.1263062367361,
            "mae_deg": 5.845556977127689,
            "valid_angle_count": 10832
          },
          "pinky": {
            "absolute_error_sum_rad": 1537.96813735889,
            "mae_deg": 8.135070466789225,
            "valid_angle_count": 10832
          },
          "ring": {
            "absolute_error_sum_rad": 1115.3588544936065,
            "mae_deg": 5.899681961321062,
            "valid_angle_count": 10832
          },
          "thumb": {
            "absolute_error_sum_rad": 1064.839354974276,
            "mae_deg": 5.6324594626531495,
            "valid_angle_count": 10832
          }
        },
        "by_hand": {
          "left": {
            "absolute_error_sum_rad": 3337.5490544303757,
            "mae_deg": 6.424359159334113,
            "valid_angle_count": 29766
          },
          "right": {
            "absolute_error_sum_rad": 3097.8581978280854,
            "mae_deg": 5.954183168921588,
            "valid_angle_count": 29810
          }
        },
        "by_hand_split": {
          "left": {
            "both": {
              "absolute_error_sum_rad": 276.5194798059383,
              "mae_deg": 7.201545066378754,
              "valid_angle_count": 2200
            },
            "gesture": {
              "absolute_error_sum_rad": 1638.5059665560839,
              "mae_deg": 5.877753355288456,
              "valid_angle_count": 15972
            },
            "user": {
              "absolute_error_sum_rad": 1422.5236080683535,
              "mae_deg": 7.02989468691036,
              "valid_angle_count": 11594
            }
          },
          "right": {
            "both": {
              "absolute_error_sum_rad": 249.58466500695795,
              "mae_deg": 6.435710142252555,
              "valid_angle_count": 2222
            },
            "gesture": {
              "absolute_error_sum_rad": 1567.5258714579613,
              "mae_deg": 5.6308850603202245,
              "valid_angle_count": 15950
            },
            "user": {
              "absolute_error_sum_rad": 1280.7476613631661,
              "mae_deg": 6.3053304362742555,
              "valid_angle_count": 11638
            }
          }
        },
        "by_joint": [
          {
            "absolute_error_sum_rad": 206.45633886333962,
            "index": 0,
            "mae_deg": 4.368196776437266,
            "name": "THUMB_CMC_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 252.52317913815205,
            "index": 1,
            "mae_deg": 5.34287754573196,
            "name": "THUMB_CMC_AA",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 288.57564913039096,
            "index": 2,
            "mae_deg": 6.105674581026398,
            "name": "THUMB_MCP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 317.2841878423933,
            "index": 3,
            "mae_deg": 6.713088947416972,
            "name": "THUMB_IP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 231.65402994747274,
            "index": 4,
            "mae_deg": 4.901328738252351,
            "name": "INDEX_MCP_AA",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 266.88148364319204,
            "index": 5,
            "mae_deg": 5.646670104484718,
            "name": "INDEX_MCP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 353.9158042225754,
            "index": 6,
            "mae_deg": 7.488139543918718,
            "name": "INDEX_PIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 210.87536460137926,
            "index": 7,
            "mae_deg": 4.461694385133483,
            "name": "INDEX_DIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 223.55420703861455,
            "index": 8,
            "mae_deg": 4.729952937853182,
            "name": "MIDDLE_MCP_AA",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 301.65601108112605,
            "index": 9,
            "mae_deg": 6.382428471085714,
            "name": "MIDDLE_MCP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 342.6658165838453,
            "index": 10,
            "mae_deg": 7.250112656446938,
            "name": "MIDDLE_PIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 237.25027153315023,
            "index": 11,
            "mae_deg": 5.019733843124921,
            "name": "MIDDLE_DIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 199.14583071759262,
            "index": 12,
            "mae_deg": 4.213521273170164,
            "name": "RING_MCP_AA",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 342.22609881625976,
            "index": 13,
            "mae_deg": 7.2408091216391215,
            "name": "RING_MCP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 383.4820558654028,
            "index": 14,
            "mae_deg": 8.113701373739895,
            "name": "RING_PIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 190.5048690943513,
            "index": 15,
            "mae_deg": 4.030696076735068,
            "name": "RING_DIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 358.62375793778483,
            "index": 16,
            "mae_deg": 7.587750281741627,
            "name": "PINKY_MCP_AA",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 512.881078623017,
            "index": 17,
            "mae_deg": 10.851521860124157,
            "name": "PINKY_MCP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 417.57535791536793,
            "index": 18,
            "mae_deg": 8.835046394835803,
            "name": "PINKY_PIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 248.88794288272038,
            "index": 19,
            "mae_deg": 5.265963330455309,
            "name": "PINKY_DIP_FE",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 259.4772594003007,
            "index": 20,
            "mae_deg": 5.490011758958086,
            "name": "wrist_pitch",
            "valid_angle_count": 2708
          },
          {
            "absolute_error_sum_rad": 289.31065738003235,
            "index": 21,
            "mae_deg": 6.121225862640782,
            "name": "wrist_yaw",
            "valid_angle_count": 2708
          }
        ],
        "by_pd_group": {
          "distal": {
            "absolute_error_sum_rad": 1204.8026359539945,
            "mae_deg": 5.09823531657315,
            "valid_angle_count": 13540
          },
          "mid": {
            "absolute_error_sum_rad": 1786.2146837175824,
            "mae_deg": 7.558534909993551,
            "valid_angle_count": 13540
          },
          "proximal": {
            "absolute_error_sum_rad": 2895.602015806551,
            "mae_deg": 6.126505711052027,
            "valid_angle_count": 27080
          }
        },
        "by_split": {
          "both": {
            "absolute_error_sum_rad": 526.1041448128963,
            "mae_deg": 6.816722542315341,
            "valid_angle_count": 4422
          },
          "gesture": {
            "absolute_error_sum_rad": 3206.031838014045,
            "mae_deg": 5.754404276134791,
            "valid_angle_count": 31922
          },
          "user": {
            "absolute_error_sum_rad": 2703.2712694315196,
            "mae_deg": 6.666926421203448,
            "valid_angle_count": 23232
          }
        },
        "by_split_user_summary": {
          "both": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 6.738172038191623,
            "std_mae_deg": 0.4941624571237397,
            "user_count": 5
          },
          "gesture": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 5.819429587851573,
            "std_mae_deg": 1.4249036883218045,
            "user_count": 36
          },
          "user": {
            "aggregation": "unweighted user means; ddof=0",
            "mean_mae_deg": 6.669391059000217,
            "std_mae_deg": 0.4387321528163526,
            "user_count": 5
          }
        },
        "by_user": {
          "user_000": {
            "absolute_error_sum_rad": 83.56809724539198,
            "mae_deg": 5.727391476195935,
            "valid_angle_count": 836
          },
          "user_001": {
            "absolute_error_sum_rad": 99.12497071968392,
            "mae_deg": 5.1631295150870455,
            "valid_angle_count": 1100
          },
          "user_002": {
            "absolute_error_sum_rad": 164.41651148995152,
            "mae_deg": 9.308668172567618,
            "valid_angle_count": 1012
          },
          "user_003": {
            "absolute_error_sum_rad": 95.19447868224233,
            "mae_deg": 6.524212752919345,
            "valid_angle_count": 836
          },
          "user_004": {
            "absolute_error_sum_rad": 108.32932391818758,
            "mae_deg": 6.561113169151957,
            "valid_angle_count": 946
          },
          "user_005": {
            "absolute_error_sum_rad": 97.84194498395664,
            "mae_deg": 4.996373000830579,
            "valid_angle_count": 1122
          },
          "user_006": {
            "absolute_error_sum_rad": 94.61737235577311,
            "mae_deg": 5.356893383995639,
            "valid_angle_count": 1012
          },
          "user_007": {
            "absolute_error_sum_rad": 111.67983295151498,
            "mae_deg": 5.288250483345346,
            "valid_angle_count": 1210
          },
          "user_008": {
            "absolute_error_sum_rad": 130.53926660535217,
            "mae_deg": 6.295748347828054,
            "valid_angle_count": 1188
          },
          "user_009": {
            "absolute_error_sum_rad": 76.65201980667189,
            "mae_deg": 4.990724120540332,
            "valid_angle_count": 880
          },
          "user_010": {
            "absolute_error_sum_rad": 127.34546829434112,
            "mae_deg": 5.922368403719792,
            "valid_angle_count": 1232
          },
          "user_011": {
            "absolute_error_sum_rad": 82.46984899601375,
            "mae_deg": 5.652122349943538,
            "valid_angle_count": 836
          },
          "user_012": {
            "absolute_error_sum_rad": 942.6258373313613,
            "mae_deg": 7.054399443452492,
            "valid_angle_count": 7656
          },
          "user_013": {
            "absolute_error_sum_rad": 68.74525755061768,
            "mae_deg": 7.161478398527724,
            "valid_angle_count": 550
          },
          "user_014": {
            "absolute_error_sum_rad": 130.16067571028543,
            "mae_deg": 6.2774893743857945,
            "valid_angle_count": 1188
          },
          "user_015": {
            "absolute_error_sum_rad": 102.79834115653648,
            "mae_deg": 4.95783761718479,
            "valid_angle_count": 1188
          },
          "user_016": {
            "absolute_error_sum_rad": 12.977623669430614,
            "mae_deg": 5.633053517935264,
            "valid_angle_count": 132
          },
          "user_017": {
            "absolute_error_sum_rad": 38.05198563449085,
            "mae_deg": 4.955041315792644,
            "valid_angle_count": 440
          },
          "user_018": {
            "absolute_error_sum_rad": 117.0042771002627,
            "mae_deg": 4.91484696687994,
            "valid_angle_count": 1364
          },
          "user_019": {
            "absolute_error_sum_rad": 114.79638838203391,
            "mae_deg": 5.435825254243951,
            "valid_angle_count": 1210
          },
          "user_020": {
            "absolute_error_sum_rad": 140.54557588358875,
            "mae_deg": 5.719224664321933,
            "valid_angle_count": 1408
          },
          "user_021": {
            "absolute_error_sum_rad": 65.5961377483618,
            "mae_deg": 8.541776921227077,
            "valid_angle_count": 440
          },
          "user_022": {
            "absolute_error_sum_rad": 100.03357705405506,
            "mae_deg": 4.65219299902116,
            "valid_angle_count": 1232
          },
          "user_023": {
            "absolute_error_sum_rad": 85.35190368100302,
            "mae_deg": 4.832316061589514,
            "valid_angle_count": 1012
          },
          "user_024": {
            "absolute_error_sum_rad": 659.6776463734277,
            "mae_deg": 6.817594692698594,
            "valid_angle_count": 5544
          },
          "user_025": {
            "absolute_error_sum_rad": 67.34700430813245,
            "mae_deg": 6.264121931339874,
            "valid_angle_count": 616
          },
          "user_026": {
            "absolute_error_sum_rad": 144.59119399782503,
            "mae_deg": 6.724403547753811,
            "valid_angle_count": 1232
          },
          "user_027": {
            "absolute_error_sum_rad": 113.72656305972487,
            "mae_deg": 4.936403092311097,
            "valid_angle_count": 1320
          },
          "user_028": {
            "absolute_error_sum_rad": 60.478804320213385,
            "mae_deg": 4.632593900462295,
            "valid_angle_count": 748
          },
          "user_029": {
            "absolute_error_sum_rad": 52.60845932923257,
            "mae_deg": 4.2815947247878805,
            "valid_angle_count": 704
          },
          "user_030": {
            "absolute_error_sum_rad": 388.44859625879326,
            "mae_deg": 6.322859410059702,
            "valid_angle_count": 3520
          },
          "user_031": {
            "absolute_error_sum_rad": 37.043130449950695,
            "mae_deg": 7.421031589981542,
            "valid_angle_count": 286
          },
          "user_032": {
            "absolute_error_sum_rad": 31.564522486180067,
            "mae_deg": 4.566954345464399,
            "valid_angle_count": 396
          },
          "user_033": {
            "absolute_error_sum_rad": 54.365869011497125,
            "mae_deg": 5.056712408960802,
            "valid_angle_count": 616
          },
          "user_034": {
            "absolute_error_sum_rad": 56.07862250925973,
            "mae_deg": 4.868285440436243,
            "valid_angle_count": 660
          },
          "user_035": {
            "absolute_error_sum_rad": 59.41968279890716,
            "mae_deg": 5.158328855126478,
            "valid_angle_count": 660
          },
          "user_036": {
            "absolute_error_sum_rad": 71.21993872424355,
            "mae_deg": 4.755946277481925,
            "valid_angle_count": 858
          },
          "user_037": {
            "absolute_error_sum_rad": 623.6433891071792,
            "mae_deg": 5.863494275857622,
            "valid_angle_count": 6094
          },
          "user_038": {
            "absolute_error_sum_rad": 669.6664410493613,
            "mae_deg": 6.976192864304499,
            "valid_angle_count": 5500
          },
          "user_039": {
            "absolute_error_sum_rad": 155.0606715234244,
            "mae_deg": 11.217578341864403,
            "valid_angle_count": 792
          }
        },
        "invalid_angle_count": 1848,
        "mode": "center",
        "overall": {
          "absolute_error_sum_rad": 6435.407252258461,
          "mae_deg": 6.1890975393160295,
          "valid_angle_count": 59576
        },
        "user_summary": {
          "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
          "mean_mae_deg": 5.944664335239466,
          "std_mae_deg": 1.3732357761195295,
          "user_count": 40
        }
      },
      "definition": "all requested labeled centers; vision_valid is the identical crop-present subset",
      "missing_by_hand": {
        "left": 53,
        "right": 51
      },
      "missing_by_split": {
        "both": 7,
        "gesture": 73,
        "user": 24
      },
      "samples": {
        "all": 2792,
        "missing_crop": 104,
        "vision_valid": 2688
      },
      "vision_valid": {
        "by_component": {
          "fingers": {
            "absolute_error_sum_rad": 5736.168137136454,
            "mae_deg": 6.113434241728766,
            "valid_angle_count": 53760
          },
          "wrist": {
            "absolute_error_sum_rad": 538.8793001844315,
            "mae_deg": 5.743212345150909,
            "valid_angle_count": 5376
          }
        },
        "by_hand": {
          "left": {
            "absolute_error_sum_rad": 3260.4284020133055,
            "mae_deg": 6.322642213495758,
            "valid_angle_count": 29546
          },
          "right": {
            "absolute_error_sum_rad": 3014.61903530758,
            "mae_deg": 5.837274334671309,
            "valid_angle_count": 29590
          }
        },
        "by_hand_split": {
          "left": {
            "both": {
              "absolute_error_sum_rad": 276.5194798059383,
              "mae_deg": 7.201545066378754,
              "valid_angle_count": 2200
            },
            "gesture": {
              "absolute_error_sum_rad": 1570.4659475837252,
              "mae_deg": 5.696446610886365,
              "valid_angle_count": 15796
            },
            "user": {
              "absolute_error_sum_rad": 1413.442974623642,
              "mae_deg": 7.011629179943844,
              "valid_angle_count": 11550
            }
          },
          "right": {
            "both": {
              "absolute_error_sum_rad": 249.58466500695795,
              "mae_deg": 6.435710142252555,
              "valid_angle_count": 2222
            },
            "gesture": {
              "absolute_error_sum_rad": 1493.4576207021819,
              "mae_deg": 5.424674689228144,
              "valid_angle_count": 15774
            },
            "user": {
              "absolute_error_sum_rad": 1271.57674959844,
              "mae_deg": 6.28393833698069,
              "valid_angle_count": 11594
            }
          }
        },
        "by_split": {
          "both": {
            "absolute_error_sum_rad": 526.1041448128963,
            "mae_deg": 6.816722542315341,
            "valid_angle_count": 4422
          },
          "gesture": {
            "absolute_error_sum_rad": 3063.923568285907,
            "mae_deg": 5.560655344106612,
            "valid_angle_count": 31570
          },
          "user": {
            "absolute_error_sum_rad": 2685.019724222082,
            "mae_deg": 6.647092037128652,
            "valid_angle_count": 23144
          }
        },
        "by_user": {
          "user_000": {
            "absolute_error_sum_rad": 83.56809724539198,
            "mae_deg": 5.727391476195935,
            "valid_angle_count": 836
          },
          "user_001": {
            "absolute_error_sum_rad": 99.12497071968392,
            "mae_deg": 5.1631295150870455,
            "valid_angle_count": 1100
          },
          "user_002": {
            "absolute_error_sum_rad": 164.41651148995152,
            "mae_deg": 9.308668172567618,
            "valid_angle_count": 1012
          },
          "user_003": {
            "absolute_error_sum_rad": 62.767252259887755,
            "mae_deg": 5.10837875869964,
            "valid_angle_count": 704
          },
          "user_004": {
            "absolute_error_sum_rad": 108.32932391818758,
            "mae_deg": 6.561113169151957,
            "valid_angle_count": 946
          },
          "user_005": {
            "absolute_error_sum_rad": 87.14001729225856,
            "mae_deg": 4.631498346515239,
            "valid_angle_count": 1078
          },
          "user_006": {
            "absolute_error_sum_rad": 94.61737235577311,
            "mae_deg": 5.356893383995639,
            "valid_angle_count": 1012
          },
          "user_007": {
            "absolute_error_sum_rad": 111.67983295151498,
            "mae_deg": 5.288250483345346,
            "valid_angle_count": 1210
          },
          "user_008": {
            "absolute_error_sum_rad": 130.53926660535217,
            "mae_deg": 6.295748347828054,
            "valid_angle_count": 1188
          },
          "user_009": {
            "absolute_error_sum_rad": 76.65201980667189,
            "mae_deg": 4.990724120540332,
            "valid_angle_count": 880
          },
          "user_010": {
            "absolute_error_sum_rad": 127.34546829434112,
            "mae_deg": 5.922368403719792,
            "valid_angle_count": 1232
          },
          "user_011": {
            "absolute_error_sum_rad": 82.46984899601375,
            "mae_deg": 5.652122349943538,
            "valid_angle_count": 836
          },
          "user_012": {
            "absolute_error_sum_rad": 942.6258373313613,
            "mae_deg": 7.054399443452492,
            "valid_angle_count": 7656
          },
          "user_013": {
            "absolute_error_sum_rad": 68.74525755061768,
            "mae_deg": 7.161478398527724,
            "valid_angle_count": 550
          },
          "user_014": {
            "absolute_error_sum_rad": 130.16067571028543,
            "mae_deg": 6.2774893743857945,
            "valid_angle_count": 1188
          },
          "user_015": {
            "absolute_error_sum_rad": 102.79834115653648,
            "mae_deg": 4.95783761718479,
            "valid_angle_count": 1188
          },
          "user_016": {
            "absolute_error_sum_rad": 12.977623669430614,
            "mae_deg": 5.633053517935264,
            "valid_angle_count": 132
          },
          "user_017": {
            "absolute_error_sum_rad": 38.05198563449085,
            "mae_deg": 4.955041315792644,
            "valid_angle_count": 440
          },
          "user_018": {
            "absolute_error_sum_rad": 117.0042771002627,
            "mae_deg": 4.91484696687994,
            "valid_angle_count": 1364
          },
          "user_019": {
            "absolute_error_sum_rad": 114.79638838203391,
            "mae_deg": 5.435825254243951,
            "valid_angle_count": 1210
          },
          "user_020": {
            "absolute_error_sum_rad": 140.54557588358875,
            "mae_deg": 5.719224664321933,
            "valid_angle_count": 1408
          },
          "user_021": {
            "absolute_error_sum_rad": 65.5961377483618,
            "mae_deg": 8.541776921227077,
            "valid_angle_count": 440
          },
          "user_022": {
            "absolute_error_sum_rad": 100.03357705405506,
            "mae_deg": 4.65219299902116,
            "valid_angle_count": 1232
          },
          "user_023": {
            "absolute_error_sum_rad": 85.35190368100302,
            "mae_deg": 4.832316061589514,
            "valid_angle_count": 1012
          },
          "user_024": {
            "absolute_error_sum_rad": 659.6776463734277,
            "mae_deg": 6.817594692698594,
            "valid_angle_count": 5544
          },
          "user_025": {
            "absolute_error_sum_rad": 67.34700430813245,
            "mae_deg": 6.264121931339874,
            "valid_angle_count": 616
          },
          "user_026": {
            "absolute_error_sum_rad": 144.59119399782503,
            "mae_deg": 6.724403547753811,
            "valid_angle_count": 1232
          },
          "user_027": {
            "absolute_error_sum_rad": 113.72656305972487,
            "mae_deg": 4.936403092311097,
            "valid_angle_count": 1320
          },
          "user_028": {
            "absolute_error_sum_rad": 60.478804320213385,
            "mae_deg": 4.632593900462295,
            "valid_angle_count": 748
          },
          "user_029": {
            "absolute_error_sum_rad": 52.60845932923257,
            "mae_deg": 4.2815947247878805,
            "valid_angle_count": 704
          },
          "user_030": {
            "absolute_error_sum_rad": 370.19705104935565,
            "mae_deg": 6.180282229987518,
            "valid_angle_count": 3432
          },
          "user_031": {
            "absolute_error_sum_rad": 21.280478580854833,
            "mae_deg": 5.038353754964998,
            "valid_angle_count": 242
          },
          "user_032": {
            "absolute_error_sum_rad": 31.564522486180067,
            "mae_deg": 4.566954345464399,
            "valid_angle_count": 396
          },
          "user_033": {
            "absolute_error_sum_rad": 54.365869011497125,
            "mae_deg": 5.056712408960802,
            "valid_angle_count": 616
          },
          "user_034": {
            "absolute_error_sum_rad": 56.07862250925973,
            "mae_deg": 4.868285440436243,
            "valid_angle_count": 660
          },
          "user_035": {
            "absolute_error_sum_rad": 59.41968279890716,
            "mae_deg": 5.158328855126478,
            "valid_angle_count": 660
          },
          "user_036": {
            "absolute_error_sum_rad": 71.21993872424355,
            "mae_deg": 4.755946277481925,
            "valid_angle_count": 858
          },
          "user_037": {
            "absolute_error_sum_rad": 623.6433891071792,
            "mae_deg": 5.863494275857622,
            "valid_angle_count": 6094
          },
          "user_038": {
            "absolute_error_sum_rad": 669.6664410493613,
            "mae_deg": 6.976192864304499,
            "valid_angle_count": 5500
          },
          "user_039": {
            "absolute_error_sum_rad": 71.8442077784348,
            "mae_deg": 6.236924072977688,
            "valid_angle_count": 660
          }
        },
        "invalid_angle_count": 0,
        "mode": "center",
        "overall": {
          "absolute_error_sum_rad": 6275.047437320885,
          "mae_deg": 6.079777705676233,
          "valid_angle_count": 59136
        },
        "user_summary": {
          "aggregation": "unweighted per-user means; population standard deviation (ddof=0)",
          "mean_mae_deg": 5.712498886926704,
          "std_mae_deg": 1.0641861940103032,
          "user_count": 40
        }
      }
    },
    "unique_rule": "source repeat1; every original hand request is consumed exactly once; both views have the same actual population",
    "unique_samples": 2792
  },
  "training": {
    "completed_epochs": 150,
    "execution_policy": {
      "batch_size_by_role": {
        "test": 200,
        "train": 200,
        "val": 200
      },
      "bitwise_training_trajectory_guaranteed": false,
      "check_val_every_n_epoch": 1,
      "early_stopping": false,
      "engineering_train_samples": null,
      "format": "original_rn18_fusion_fit_policy_v1",
      "full_paper_complete": false,
      "gradient_clip_val": 1,
      "loader": {
        "persistent_workers": true,
        "pin_memory": true,
        "prefetch_factor": 2
      },
      "logger": "original TensorBoardLogger; owned output path/version only",
      "max_epochs": 150,
      "monitor": "val_mae",
      "monitor_mode": "min",
      "native_global_loader_rng": true,
      "num_sanity_val_steps": 2,
      "optimizer_class": "torch.optim.adam.Adam",
      "optimizer_lr": 0.0006,
      "optimizer_weight_decay": 0.0001,
      "precision": "bf16-mixed",
      "protocol_sha256": "fa505397b72c600bea9987d8a96a41d324e9064cb1cf8b83c876867462362ac9",
      "save_last": true,
      "save_top_k": 3,
      "scheduler_T_max": 150,
      "scheduler_eta_min": 1e-05,
      "scientific_eligible": false,
      "scope": "full",
      "sha256": "ce2a9448114a70a2dd80381aa36132a07e6a79c2a660cd69eb7b29ef0d763bb2",
      "single_rank": true,
      "workers": 10,
      "workers_by_role": {
        "test": 10,
        "train": 10,
        "val": 10
      }
    },
    "fit_result_sha256": "bb4aa8f8c67cb45e9c5917a33282364e869bbfef1907e41105804fb35cb3d15e",
    "full_training_terminated": true,
    "optimizer_executions": 46950,
    "scope": "full",
    "selected_best": {
      "checkpoint_sha256": "b37f0f3c36ce48ecdcba43428db5529195bc08abfe7c7f5c06d5d4842b91dcd2",
      "file": "rn18-s-8ch-centerfusion-epoch=144-val_mae=0.1063.ckpt",
      "score": 0.10630998760461807,
      "validated": true
    }
  },
  "validation_profile_sha256": "6c9a33d452e6a7d75a0cc71c1d10e5a8835b2a38aabb2ca9faa2af260368b1d8",
  "variant": "fusion_rn18_s_center_8ch"
}
```

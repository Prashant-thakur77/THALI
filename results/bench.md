_measured on i7-13650HX CPU + UHD iGPU; same IR runs on Core Ultra NPU with -d NPU and static shapes, not measured here._

| skill | precision | device | size MB | p50 ms | p95 ms | infer/s |
|---|---|---|---|---|---|---|
| open_drawer | fp32 | CPU | 132.7 | 47.63 | 52.42 | 20.8 |
| open_drawer | fp32 | GPU | 132.7 | 282.97 | 285.06 | 3.5 |
| open_drawer | fp16 | CPU | 67.1 | 49.46 | 51.48 | 20.1 |
| open_drawer | fp16 | GPU | 67.1 | 282.76 | 286.5 | 3.5 |
| open_drawer | int8 | CPU | 34.9 | 16.88 | 17.59 | 59.0 |
| open_drawer | int8 | GPU | 34.9 | 202.16 | 202.53 | 4.9 |
| pick_place_fork | fp32 | CPU | 132.7 | 49.09 | 50.4 | 20.3 |
| pick_place_fork | fp32 | GPU | 132.7 | 284.32 | 287.34 | 3.5 |
| pick_place_fork | fp16 | CPU | 67.1 | 49.23 | 51.04 | 20.2 |
| pick_place_fork | fp16 | GPU | 67.1 | 282.25 | 286.39 | 3.5 |
| pick_place_fork | int8 | CPU | 34.9 | 17.07 | 17.97 | 58.1 |
| pick_place_fork | int8 | GPU | 34.9 | 202.18 | 202.6 | 4.9 |
| pick_place_plate | fp32 | CPU | 132.7 | 49.44 | 53.33 | 20.0 |
| pick_place_plate | fp32 | GPU | 132.7 | 282.22 | 285.61 | 3.5 |
| pick_place_plate | fp16 | CPU | 67.1 | 49.32 | 51.17 | 20.2 |
| pick_place_plate | fp16 | GPU | 67.1 | 282.24 | 285.09 | 3.5 |
| pick_place_plate | int8 | CPU | 34.9 | 17.09 | 17.95 | 58.2 |
| pick_place_plate | int8 | GPU | 34.9 | 202.63 | 203.16 | 4.9 |
| pick_place_mug | fp32 | CPU | 132.7 | 49.19 | 50.47 | 20.3 |
| pick_place_mug | fp32 | GPU | 132.7 | 283.51 | 286.85 | 3.5 |
| pick_place_mug | fp16 | CPU | 67.1 | 49.15 | 50.97 | 20.2 |
| pick_place_mug | fp16 | GPU | 67.1 | 282.92 | 285.52 | 3.5 |
| pick_place_mug | int8 | CPU | 34.9 | 16.92 | 17.71 | 58.8 |
| pick_place_mug | int8 | GPU | 34.9 | 201.69 | 203.42 | 4.9 |
| handoff_spoon | fp32 | CPU | 132.7 | 49.39 | 51.12 | 20.2 |
| handoff_spoon | fp32 | GPU | 132.7 | 283.23 | 286.45 | 3.5 |
| handoff_spoon | fp16 | CPU | 67.1 | 49.37 | 51.24 | 20.1 |
| handoff_spoon | fp16 | GPU | 67.1 | 282.93 | 286.09 | 3.5 |
| handoff_spoon | int8 | CPU | 34.9 | 17.14 | 18.01 | 58.0 |
| handoff_spoon | int8 | GPU | 34.9 | 201.34 | 201.64 | 5.0 |
| hold_mug | fp32 | CPU | 132.7 | 49.42 | 51.04 | 20.1 |
| hold_mug | fp32 | GPU | 132.7 | 283.34 | 286.44 | 3.5 |
| hold_mug | fp16 | CPU | 67.1 | 49.89 | 54.67 | 19.8 |
| hold_mug | fp16 | GPU | 67.1 | 283.46 | 286.16 | 3.5 |
| hold_mug | int8 | CPU | 34.9 | 17.04 | 17.66 | 58.5 |
| hold_mug | int8 | GPU | 34.9 | 201.38 | 201.85 | 5.0 |
| pour | fp32 | CPU | 132.7 | 49.75 | 52.67 | 20.0 |
| pour | fp32 | GPU | 132.7 | 283.06 | 286.05 | 3.5 |
| pour | fp16 | CPU | 67.1 | 50.16 | 57.04 | 19.6 |
| pour | fp16 | GPU | 67.1 | 283.17 | 285.62 | 3.5 |
| pour | int8 | CPU | 34.9 | 17.31 | 18.38 | 57.2 |
| pour | int8 | GPU | 34.9 | 201.55 | 203.21 | 5.0 |

VLM planner (Qwen/Qwen2-VL-2B-Instruct INT4 (optimum-cli export, group 128)) on CPU: **44.6 tok/s**, TTFT 1130 ms, mean plan 5.41 s. iGPU: loads and answers once, then crashes in the GPU plugin (docs/BLOCKERS.md).

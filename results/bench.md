_measured on i7-13650HX CPU + UHD iGPU; same IR runs on Core Ultra NPU with -d NPU and static shapes, not measured here._

| skill | precision | device | size MB | p50 ms | p95 ms | infer/s |
|---|---|---|---|---|---|---|
| open_drawer | fp32 | CPU | 132.7 | 108.43 | 119.29 | 9.1 |
| open_drawer | fp32 | GPU | 132.7 | 553.14 | 556.99 | 1.8 |
| open_drawer | fp16 | CPU | 67.1 | 108.14 | 124.07 | 9.1 |
| open_drawer | fp16 | GPU | 67.1 | 554.16 | 558.49 | 1.8 |
| open_drawer | int8 | CPU | 34.9 | 36.56 | 38.6 | 27.6 |
| open_drawer | int8 | GPU | 34.9 | 416.53 | 434.23 | 2.4 |
| pick_place_fork | fp32 | CPU | 132.7 | 109.79 | 119.0 | 9.1 |
| pick_place_fork | fp32 | GPU | 132.7 | 555.62 | 557.33 | 1.8 |
| pick_place_fork | fp16 | CPU | 67.1 | 115.83 | 129.58 | 8.7 |
| pick_place_fork | fp16 | GPU | 67.1 | 554.37 | 558.05 | 1.8 |
| pick_place_fork | int8 | CPU | 34.9 | 36.57 | 41.91 | 27.3 |
| pick_place_fork | int8 | GPU | 34.9 | 415.67 | 434.47 | 2.4 |
| pick_place_plate | fp32 | CPU | 132.7 | 114.58 | 132.82 | 8.6 |
| pick_place_plate | fp32 | GPU | 132.7 | 555.36 | 558.59 | 1.8 |
| pick_place_plate | fp16 | CPU | 67.1 | 109.28 | 119.66 | 9.1 |
| pick_place_plate | fp16 | GPU | 67.1 | 554.47 | 558.41 | 1.8 |
| pick_place_plate | int8 | CPU | 34.9 | 36.54 | 39.48 | 27.5 |
| pick_place_plate | int8 | GPU | 34.9 | 415.01 | 434.47 | 2.4 |
| pick_place_mug | fp32 | CPU | 132.7 | 111.58 | 120.22 | 9.0 |
| pick_place_mug | fp32 | GPU | 132.7 | 555.88 | 558.91 | 1.8 |
| pick_place_mug | fp16 | CPU | 67.1 | 111.02 | 119.26 | 9.0 |
| pick_place_mug | fp16 | GPU | 67.1 | 555.52 | 559.98 | 1.8 |
| pick_place_mug | int8 | CPU | 34.9 | 36.49 | 44.37 | 26.9 |
| pick_place_mug | int8 | GPU | 34.9 | 414.79 | 434.84 | 2.4 |
| handoff_spoon | fp32 | CPU | 132.7 | 110.91 | 124.45 | 9.0 |
| handoff_spoon | fp32 | GPU | 132.7 | 555.42 | 559.85 | 1.8 |
| handoff_spoon | fp16 | CPU | 67.1 | 110.53 | 126.84 | 8.9 |
| handoff_spoon | fp16 | GPU | 67.1 | 554.15 | 557.59 | 1.8 |
| handoff_spoon | int8 | CPU | 34.9 | 37.12 | 46.98 | 26.3 |
| handoff_spoon | int8 | GPU | 34.9 | 416.0 | 434.64 | 2.4 |
| hold_mug | fp32 | CPU | 132.7 | 109.34 | 125.87 | 8.9 |
| hold_mug | fp32 | GPU | 132.7 | 553.98 | 558.35 | 1.8 |
| hold_mug | fp16 | CPU | 67.1 | 110.13 | 121.6 | 9.0 |
| hold_mug | fp16 | GPU | 67.1 | 555.49 | 558.41 | 1.8 |
| hold_mug | int8 | CPU | 34.9 | 36.96 | 43.45 | 26.9 |
| hold_mug | int8 | GPU | 34.9 | 416.5 | 434.28 | 2.4 |
| pour | fp32 | CPU | 132.7 | 110.52 | 119.57 | 9.1 |
| pour | fp32 | GPU | 132.7 | 554.56 | 557.6 | 1.8 |
| pour | fp16 | CPU | 67.1 | 108.13 | 119.31 | 9.2 |
| pour | fp16 | GPU | 67.1 | 556.27 | 557.81 | 1.8 |
| pour | int8 | CPU | 34.9 | 37.11 | 40.63 | 27.1 |
| pour | int8 | GPU | 34.9 | 414.81 | 434.16 | 2.4 |

VLM planner (Qwen/Qwen2-VL-2B-Instruct INT4 (optimum-cli export, group 128)) on CPU: **44.6 tok/s**, TTFT 1130 ms, mean plan 5.41 s. iGPU: loads and answers once, then crashes in the GPU plugin (docs/BLOCKERS.md).

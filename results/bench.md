_measured on i7-13650HX CPU + UHD iGPU; same IR runs on Core Ultra NPU with -d NPU and static shapes, not measured here._

| skill | precision | device | size MB | p50 ms | p95 ms | infer/s |
|---|---|---|---|---|---|---|
| open_drawer | fp32 | CPU | 132.7 | 114.39 | 131.6 | 8.7 |
| open_drawer | fp32 | GPU | 132.7 | 555.63 | 557.92 | 1.8 |
| open_drawer | fp16 | CPU | 67.1 | 111.51 | 127.86 | 8.8 |
| open_drawer | fp16 | GPU | 67.1 | 556.94 | 559.05 | 1.8 |
| open_drawer | int8 | CPU | 34.9 | 36.95 | 40.16 | 27.3 |
| open_drawer | int8 | GPU | 34.9 | 421.94 | 430.42 | 2.4 |

VLM planner (Qwen/Qwen2-VL-2B-Instruct INT4 (optimum-cli export, group 128)) on CPU: **44.6 tok/s**, TTFT 1130 ms, mean plan 5.41 s. iGPU: loads and answers once, then crashes in the GPU plugin (docs/BLOCKERS.md).

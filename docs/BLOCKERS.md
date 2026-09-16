# Blockers and deviations

Things that failed twice or are infeasible on this machine, with what was done instead.

## EGL offscreen rendering (Phase 1)
`MUJOCO_GL=egl` fails on this Dell G15 (Wayland session, NVIDIA + Intel hybrid): `eglQueryDevicesEXT`
enumeration is broken in the installed PyOpenGL/EGL stack and every vendor-file / device-id variant
throws `EGLError`. `MUJOCO_GL=glfw` renders correctly against the local display (`:0`), so the env
sets `MUJOCO_GL=glfw` by default (`souschef_env/__init__.py`) and every Makefile target exports it.
**Consequence:** demo recording and ACT training must run on a machine with a display session
(this laptop is fine); the Kaggle notebook uses EGL as normal. Shadows/reflections are disabled in
the env renderer (20 → 6 ms per 240×320 frame).

## Intel iGPU for the VLM planner (Phase 4)
`openvino_genai.VLMPipeline(..., "GPU")` loads and answers once on the Raptor Lake UHD iGPU (9 tok/s, but a
24–55 s time-to-first-token for a 240×320 frame + ~700-token prompt), then the second `generate` dies inside the
GPU plugin (`clWaitForEvents -9999` / `CL_INVALID_VALUE` in `ocl_memory.cpp`), whether or not chat mode is used.
The CPU is stable and much faster on this box (44 tok/s, 1.1 s TTFT, `results/planner_eval.json`), so the
planner defaults to `device="CPU"`. The bench (Phase 8) still measures the small policy IRs on GPU, where the
plugin is fine. On a Core Ultra the same IR would run on the NPU/iGPU with `-d NPU`/`GPU`; not measured here.

## optimum-cli export staging in /tmp (Phase 4)
`optimum-cli export openvino` stages each sub-model under `$TMPDIR` before writing the output directory; with
/tmp a 7.5 GB tmpfs the 1.3 GB vision merger failed with `basic_ios::clear: iostream error`. `planner/export.sh`
now sets `TMPDIR=$HOME/tmp`.

## Speechmatics free-tier concurrency (Phase 6)
Back-to-back realtime sessions hit `4005 quota_exceeded` when a session is opened within a few seconds of the
previous one closing. `voice/listen.transcribe_file` retries with backoff and `eval/voice_test.py` pauses 5 s
between files.

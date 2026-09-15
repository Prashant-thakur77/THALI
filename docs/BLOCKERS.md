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

# Handoff changelog — what changed each push (newest on top)

Aayush's agent appends a line when it pushes a model/insight change. Robert's agent reads the top to see what's new,
then notes back what it built on the frontend.

---

## 2026-06-10 — handoff boundary established
- Created `handoff/` as the model↔frontend boundary (`capabilities.json` + `samples/` + this log).
- Seeded `capabilities.json` with the current Mary/Qualia capability set (lenses: attention, emotion, memory, buy_sell,
  manipulation). `sycophancy` marked PREVIEW (not live in serve.py yet).
- Frontend: `/ops/components` Component & API Library rebuilt (Mary/Qualia/All + per-lens + API JSON).
- **Next for Aayush's agent:** when you change a model, update `capabilities.json`, drop a real sample in
  `samples/<model>/`, and add a line here.

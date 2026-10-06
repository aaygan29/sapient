# semantic-demo/

Demonstrates the semantic layer (`neurosignal/semantic.py`): any text → linked concepts →
construct loadings → idea trajectory over time, under an audience lens, feeding the neuro engine.

```bash
python3 run_semantic_demo.py
```
Outputs `figures/concept_graph.png` (linked concepts, colored by reward vs conflict loading),
`figures/idea_trajectory.png` (how reward/conflict/attention move through a speech), and
`semantic_result.json`.

**Honest framing:** a predicted semantic-affective map grounded in Warriner et al. (2013) affect
norms — aggregate/population read-out, NOT mind-reading and NOT "neurolinguistic programming."
Full rationale, the Huth/Tang brain-semantic path, and the investor-grade validation bar:
[`../../neurosignal/SEMANTIC_LAYER.md`](../../neurosignal/SEMANTIC_LAYER.md).

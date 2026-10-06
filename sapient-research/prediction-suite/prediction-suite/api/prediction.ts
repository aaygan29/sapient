/**
 * prediction.ts — API surface for the Prediction Suite.
 *
 * Thin typed contract between the Python engine (ag_push/engine) and the frontend
 * (ag_push/frontend/PredictionSuite.tsx). Wire this into server.ts as a route, or
 * port the handler to a Vercel serverless function. The engine is invoked out of
 * process (subprocess to `python engine/serve_forecast.py`, or via the sapient-serving
 * model pool) — kept deliberately transport-agnostic so it drops into the existing
 * server without assuming a framework.
 */

export type ForecastRow = {
  adId: string;
  index: number;            // 0-100 calibrated market-forecast index
  band: [number, number];   // 68% uncertainty interval
  driver: string;           // dominant construct/region
  sampleRobust: boolean;    // affect-driven (generalizes) vs integrative (sample-sensitive)
  provenance: string;       // plain-language "why this number"
};

export type SuitePayload = {
  ranking: ForecastRow[];
  brainR: number;
  selfReportR: number;
  passes: boolean;
  trajectory: { peakS: number; dropS: number; summary: string; recommendations: string[] };
  persona: { topConstructs: string[]; confidence: number };
  disclaimer: string;
};

export type ForecastRequest = {
  /** one entry per ad: the 20,484-vertex Mary prediction, or a handle the model pool resolves */
  ads: Array<{ adId: string; vmapRef: string; selfReport?: number }>;
  construct?: 'value' | 'reward' | 'emotion' | 'arousal' | 'attention' | 'memory';
  /** optional measured outcomes to calibrate + validate (CTR / sales lift / recall) */
  outcomes?: number[];
};

/**
 * Invoke the engine. Replace the body with a subprocess call or a fetch to the
 * sapient-serving pool. Shape is fixed so the frontend can be built against it now.
 */
export async function runForecast(_req: ForecastRequest): Promise<SuitePayload> {
  // Reference implementation (out-of-process):
  //   const out = await execFileP('python', ['engine/serve_forecast.py'], { input: JSON.stringify(_req) });
  //   return JSON.parse(out.stdout);
  throw new Error(
    'runForecast: wire to engine (subprocess `python engine/serve_forecast.py` or ' +
    'sapient-serving pool). See ag_push/engine/behavioral_bridge.py for the contract.'
  );
}

/** Honesty boundary enforced server-side too: never emit a claim the gate rejected. */
export const DISCLAIMER =
  'Forecasts response, not persuasion. Construct claims are withheld when the ' +
  'specificity gate attributes the signal to low-level sensory confounds. Grounded in ' +
  'public neuroforecasting research (Knutson 2007; Falk 2012/2016; Kühn 2016; Genevsky 2025).';

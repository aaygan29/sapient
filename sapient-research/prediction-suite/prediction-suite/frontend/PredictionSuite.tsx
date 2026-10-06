/**
 * PredictionSuite.tsx — drop-in panel for the Sapient app.
 *
 * Renders the four Prediction-Suite capabilities (PredictiveValue, BrainTrajectory,
 * PersonaMap, BehavioralBridge) against the /api/prediction endpoint (see
 * ag_push/api/prediction.ts). Falls back to the illustrative demo payload when no
 * live result is passed, so it renders immediately inside the app shell.
 *
 * Stack parity: Tailwind v4 utility classes, lucide-react icons, motion/react —
 * all already in the sapient package.json. No new dependencies.
 *
 * Wire-in: add a route/tab in App.tsx ->  <PredictionSuite forecast={result} />
 */
import { useMemo } from 'react';
import { motion } from 'motion/react';
import { TrendingUp, Activity, Fingerprint, Brain, ArrowUpRight } from 'lucide-react';

export type ForecastRow = {
  adId: string;
  index: number;          // 0-100 market-forecast index
  band: [number, number]; // 68% interval
  driver: string;         // dominant evidence
  sampleRobust: boolean;
};

export type SuitePayload = {
  ranking: ForecastRow[];
  brainR: number;         // brain vs outcome
  selfReportR: number;    // self-report vs outcome
  passes: boolean;
  trajectory: { peakS: number; dropS: number; summary: string };
  persona: { topConstructs: string[]; confidence: number };
};

// Illustrative fallback — mirrors ag_push/engine/demo.py output. Labelled as such in UI.
const DEMO: SuitePayload = {
  ranking: [
    { adId: 'ad03', index: 72.3, band: [68.6, 75.9], driver: 'value (affect)', sampleRobust: true },
    { adId: 'ad34', index: 70.1, band: [66.5, 73.8], driver: 'value (affect)', sampleRobust: true },
    { adId: 'ad08', index: 69.0, band: [65.4, 72.7], driver: 'value (affect)', sampleRobust: true },
    { adId: 'ad17', index: 51.4, band: [39.9, 62.9], driver: 'attention', sampleRobust: false },
  ],
  brainR: 0.95,
  selfReportR: 0.63,
  passes: true,
  trajectory: { peakS: 15, dropS: 18, summary: 'Peak arousal at :15s · memory encoding :15–18s' },
  persona: { topConstructs: ['value', 'attention', 'memory'], confidence: 1.0 },
};

const CAPS = [
  { key: 'pv', tag: '01 · PredictiveValue', icon: TrendingUp, title: 'Forecast the outcome',
    body: 'Affect-weighted brain response → real outcomes (CTR, sales, recall), validated leave-one-ad-out. The brain term must beat self-report to pass.',
    q: 'Will this campaign actually move the market?' },
  { key: 'bt', tag: '02 · BrainTrajectory', icon: Activity, title: 'See the second-by-second arc',
    body: 'Engagement, arousal and memory-encoding across the runtime, with the exact frames where attention peaks and drops.',
    q: 'Where do I lose them, and where do they remember?' },
  { key: 'pm', tag: '03 · PersonaMap', icon: Fingerprint, title: 'Phenotype the audience',
    body: 'How an individual brain deviates from the population — who a message is really for — transferable from 1–3 examples.',
    q: 'Which segment is this actually for?' },
  { key: 'bb', tag: '04 · BehavioralBridge', icon: Brain, title: 'Weight what generalizes',
    body: 'Up-weights the anticipatory-affect signal (NAcc, mOFC) that generalizes to aggregate choice; stays valid on non-representative samples.',
    q: 'Why trust a forecast from a small panel?' },
];

export default function PredictionSuite({ forecast }: { forecast?: SuitePayload }) {
  const data = forecast ?? DEMO;
  const isDemo = !forecast;
  const maxIdx = useMemo(() => Math.max(...data.ranking.map((r) => r.index), 1), [data]);

  return (
    <div className="mx-auto max-w-5xl px-6 py-14 text-zinc-100">
      <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-zinc-500">
        Sapient · Prediction Suite {isDemo && '· illustrative'}
      </div>
      <h1 className="mt-3 text-4xl font-semibold leading-tight tracking-tight">
        From predicted brain response to{' '}
        <span className="bg-gradient-to-r from-emerald-300 to-blue-400 bg-clip-text text-transparent">
          forecast behavior
        </span>.
      </h1>
      <p className="mt-3 max-w-2xl text-[17px] text-zinc-400">
        Mary predicts how a brain responds. This layer turns that into a{' '}
        <span className="text-zinc-100">calibrated forecast of what the market will do</span> — grounded in the
        neuroforecasting paradigm that predicted a 400,000-person campaign from 50 brains, and gated so it never
        reports a number the signal can't carry.
      </p>

      {/* capability cards */}
      <div className="mt-10 grid grid-cols-1 gap-4 sm:grid-cols-2">
        {CAPS.map((c, i) => (
          <motion.div key={c.key}
            initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05, duration: 0.4 }}
            className="rounded-2xl border border-zinc-800 bg-zinc-950 p-5">
            <div className="flex items-center gap-2 font-mono text-[10.5px] uppercase tracking-[0.1em] text-emerald-300">
              <c.icon className="h-3.5 w-3.5" /> {c.tag}
            </div>
            <h3 className="mt-2 text-lg font-semibold tracking-tight">{c.title}</h3>
            <p className="mt-1.5 text-sm text-zinc-400">{c.body}</p>
            <p className="mt-2.5 text-[13px] italic text-zinc-500">"{c.q}"</p>
          </motion.div>
        ))}
      </div>

      {/* validation strip */}
      <div className="mt-4 flex flex-wrap items-center gap-3 rounded-2xl border border-zinc-800 bg-zinc-950 p-5">
        <span className="font-mono text-2xl font-semibold">r = {data.brainR.toFixed(2)}</span>
        <span className="text-xs text-zinc-500">brain vs outcome</span>
        <span className={`rounded-full border px-2 py-0.5 font-mono text-[10px] ${
          data.passes ? 'border-emerald-400/40 text-emerald-300' : 'border-red-400/40 text-red-300'}`}>
          {data.passes ? 'PASS' : 'FAIL'} · beats self-report r={data.selfReportR.toFixed(2)}
        </span>
      </div>

      {/* forecast ranking */}
      <div className="mt-10 font-mono text-[11px] uppercase tracking-[0.14em] text-zinc-500">
        Campaign forecast · ranked {isDemo && '(illustrative)'}
      </div>
      <div className="mt-3 overflow-hidden rounded-2xl border border-zinc-800">
        <table className="w-full text-sm">
          <thead>
            <tr className="font-mono text-[10.5px] uppercase tracking-wider text-zinc-500">
              <th className="px-4 py-2.5 text-left">Ad</th>
              <th className="px-4 py-2.5 text-left">Index</th>
              <th className="px-4 py-2.5 text-left">Relative</th>
              <th className="px-4 py-2.5 text-left">Driver</th>
              <th className="px-4 py-2.5 text-left">Band</th>
            </tr>
          </thead>
          <tbody>
            {data.ranking.map((r) => (
              <tr key={r.adId} className="border-t border-zinc-800/70">
                <td className="px-4 py-2.5 font-mono">{r.adId}</td>
                <td className="px-4 py-2.5 font-mono">{r.index.toFixed(1)}</td>
                <td className="px-4 py-2.5">
                  <div className="h-1.5 rounded-full bg-gradient-to-r from-blue-400 to-emerald-300"
                    style={{ width: `${(r.index / maxIdx) * 100}%` }} />
                </td>
                <td className="px-4 py-2.5 text-zinc-300">
                  {r.driver}{!r.sampleRobust && <span className="ml-1 text-zinc-500">· sample-sensitive</span>}
                </td>
                <td className="px-4 py-2.5 font-mono text-zinc-500">{r.band[0].toFixed(1)}–{r.band[1].toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-8 text-[12.5px] text-zinc-500">
        {isDemo && 'Figures are produced by ag_push/engine/demo.py on synthetic data and labelled illustrative. '}
        The suite forecasts and explains response; it makes no claim to change minds. Grounded in Knutson (2007),
        Falk (2012, 2016), Kühn (2016), Genevsky (2025).{' '}
        <a href="/ag_push/RESEARCH_BASIS.md" className="inline-flex items-center gap-0.5 text-zinc-300 underline">
          evidence <ArrowUpRight className="h-3 w-3" />
        </a>
      </p>
    </div>
  );
}

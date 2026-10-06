"""
BrainTrajectory™ — Temporal Brain Dynamics During Stimulus Presentation

Models how brain state evolves over the duration of an ad (0–30s).
Captures phase transitions, attention peaks, memory encoding moments.
Enables creative optimization ("add emotional beat at :15 to match peak arousal").

Based on BCNE (unsupervised manifold learning of dynamic brain data, 2508.11672).
Returns: temporal heatmap of engagement/arousal/memory over time + identified events.

Usage:
    trajectory = BrainTrajectory.from_timeseries(vmaps_t, t_samples, construct='arousal')
    heatmap = trajectory.timeline_heatmap()  # (n_times, 1) engagement over time
    events = trajectory.detect_phase_transitions()  # [{time, type, strength}]
    trajectory.plot_timeline('output.png')
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy import stats, signal
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from typing import Optional


@dataclass
class BrainEvent:
    """Identified event in brain-state trajectory."""
    time_s: float                   # Event time in seconds
    event_type: str                 # 'attention_peak', 'arousal_rise', 'memory_window', 'attention_dip'
    magnitude: float                # Normalized strength (0–1)
    construct: str                  # Which dimension (arousal, attention, memory, etc.)
    window_s: tuple                 # (start, end) time window of event


class BrainTrajectory:
    """
    Models brain-state evolution during stimulus (e.g., 30s ad).
    Input: predicted fMRI maps at multiple timepoints during the stimulus.
    Output: engagement timeline, identified events, recommendations.
    """

    def __init__(self, vmaps_t: np.ndarray, t_samples: np.ndarray,
                 construct: str = 'arousal', dt: float = 0.5):
        """
        Args:
            vmaps_t: (n_timepoints, 20484) predicted fMRI vertices over time
            t_samples: (n_timepoints,) time in seconds for each sample (e.g., [0, 1, 2, ..., 30])
            construct: 'arousal', 'attention', 'memory', or custom latent dimension
            dt: temporal resolution for smoothing (seconds)
        """
        assert vmaps_t.shape[0] == len(t_samples)
        self.vmaps_t = vmaps_t
        self.t_samples = t_samples
        self.construct = construct
        self.dt = dt
        self.n_vertices = vmaps_t.shape[1]
        self.n_timepoints = vmaps_t.shape[0]

    @staticmethod
    def from_timeseries(video_frames: np.ndarray,
                       fps: float = 2.0,
                       construct: str = 'arousal') -> BrainTrajectory:
        """Convenience: create from video frames + fMRI predictions at video fps."""
        n_frames = video_frames.shape[0]
        t_samples = np.arange(n_frames) / fps  # seconds
        return BrainTrajectory(video_frames, t_samples, construct=construct, dt=1.0 / fps)

    def compute_engagement_timeline(self, construct_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project brain states onto a single construct dimension over time.

        Args:
            construct_weights: (20484,) spatial weights for the construct (e.g., from mary_readouts)

        Returns:
            (t_samples, engagement_score): time and normalized engagement [0–1]
        """
        # Project each timepoint onto construct
        raw_scores = np.array([np.dot(v, construct_weights) for v in self.vmaps_t])

        # Normalize to [0, 1]
        vmin, vmax = np.percentile(raw_scores, [2, 98])
        engagement = np.clip((raw_scores - vmin) / (vmax - vmin + 1e-8), 0, 1)

        # Smooth (Gaussian, preserve peaks)
        sigma = self.dt / (self.t_samples[1] - self.t_samples[0]) if len(self.t_samples) > 1 else 1
        engagement_smooth = gaussian_filter1d(engagement, sigma=max(sigma, 0.5))

        return self.t_samples, engagement_smooth

    def detect_phase_transitions(self, construct_weights: np.ndarray,
                                min_duration_s: float = 1.0) -> list[BrainEvent]:
        """
        Identify significant changes in brain state (attention rise/fall, arousal peak, etc.).
        Uses derivative + threshold to find transitions.

        Args:
            construct_weights: (20484,) spatial weights
            min_duration_s: minimum event duration (seconds)

        Returns:
            List of BrainEvent objects with time, type, magnitude.
        """
        t, engagement = self.compute_engagement_timeline(construct_weights)

        # Compute derivative (rate of change)
        dt_array = np.diff(t)
        deng = np.diff(engagement) / dt_array  # d(engagement)/dt

        # Standardize for threshold
        deng_norm = (deng - np.mean(deng)) / (np.std(deng) + 1e-8)

        events = []
        min_idx = int(min_duration_s / np.mean(dt_array))  # min samples

        # Find peaks in derivative (rising phase)
        peaks_rise, _ = signal.find_peaks(deng_norm, height=1.0, distance=min_idx)
        for idx in peaks_rise:
            if idx < len(t) - 1:
                events.append(BrainEvent(
                    time_s=float(t[idx]),
                    event_type='arousal_rise' if self.construct == 'arousal' else 'attention_rise',
                    magnitude=float(np.clip(deng_norm[idx] / 3.0, 0, 1)),  # normalized
                    construct=self.construct,
                    window_s=(float(t[max(0, idx - min_idx)]), float(t[min(len(t) - 1, idx + min_idx)]))
                ))

        # Find peaks in engagement level (absolute maximum)
        peaks_engagement, props = signal.find_peaks(engagement, height=0.6, distance=min_idx)
        for idx in peaks_engagement:
            if idx not in peaks_rise:  # avoid duplicates
                events.append(BrainEvent(
                    time_s=float(t[idx]),
                    event_type='peak_' + self.construct,
                    magnitude=float(engagement[idx]),
                    construct=self.construct,
                    window_s=(float(t[max(0, idx - min_idx)]), float(t[min(len(t) - 1, idx + min_idx)]))
                ))

        # Find dips (falling phase)
        dips, _ = signal.find_peaks(-deng_norm, height=1.0, distance=min_idx)
        for idx in dips:
            if idx < len(t) - 1:
                events.append(BrainEvent(
                    time_s=float(t[idx]),
                    event_type='attention_dip',
                    magnitude=float(np.clip(abs(deng_norm[idx]) / 3.0, 0, 1)),
                    construct=self.construct,
                    window_s=(float(t[max(0, idx - min_idx)]), float(t[min(len(t) - 1, idx + min_idx)]))
                ))

        return sorted(events, key=lambda e: e.time_s)

    def estimate_memory_encoding_window(self, construct_weights: np.ndarray,
                                       threshold: float = 0.7) -> Optional[BrainEvent]:
        """
        Heuristic: identify time window where sustained engagement/arousal above threshold.
        This often correlates with memory encoding (Boksem & Smidts).

        Args:
            construct_weights: (20484,) spatial weights
            threshold: engagement level (0–1) above which encoding occurs

        Returns:
            BrainEvent or None if no sustained period.
        """
        t, engagement = self.compute_engagement_timeline(construct_weights)

        # Find sustained engagement
        above_thresh = engagement > threshold
        if not np.any(above_thresh):
            return None

        # Find longest contiguous segment
        segments = np.split(np.where(above_thresh)[0], np.where(np.diff(np.where(above_thresh)[0]) > 1)[0] + 1)
        longest = max(segments, key=len)

        start_idx, end_idx = longest[0], longest[-1]
        start_t, end_t = t[start_idx], t[end_idx]

        return BrainEvent(
            time_s=(start_t + end_t) / 2.0,
            event_type='memory_encoding_window',
            magnitude=float(np.mean(engagement[longest])),
            construct=self.construct,
            window_s=(float(start_t), float(end_t))
        )

    def timeline_heatmap(self, construct_weights: np.ndarray, plot: bool = False) -> dict:
        """
        Generate heatmap data: time vs engagement.

        Returns:
            {time_s: [...], engagement: [...], events: [...]}
        """
        t, eng = self.compute_engagement_timeline(construct_weights)
        events = self.detect_phase_transitions(construct_weights)
        mem_window = self.estimate_memory_encoding_window(construct_weights)
        if mem_window:
            events.append(mem_window)

        return {
            'time_s': t.tolist(),
            'engagement': eng.tolist(),
            'events': events,
            'construct': self.construct,
            'summary': self._summarize_events(events)
        }

    @staticmethod
    def _summarize_events(events: list[BrainEvent]) -> str:
        """Human-readable summary of detected events."""
        if not events:
            return "No significant events detected."

        key_events = [e for e in events if e.event_type in ['peak_' + e.construct, 'memory_encoding_window']]
        if key_events:
            return (f"Peak {key_events[0].construct} at :{key_events[0].time_s:.0f}s. "
                   f"{'Memory encoding likely at ' + f':{key_events[-1].time_s:.0f}–{key_events[-1].window_s[1]:.0f}s' if any(e.event_type == 'memory_encoding_window' for e in key_events) else ''}")
        return f"Detected {len(events)} significant transitions."

    def creative_recommendations(self, construct_weights: np.ndarray) -> list[str]:
        """
        Generate actionable recommendations for content editors.
        E.g., "Add emotional beat at :15 to match attention peak."
        """
        t, eng = self.compute_engagement_timeline(construct_weights)
        events = self.detect_phase_transitions(construct_weights)

        recs = []

        # Identify attention dips
        dips = [e for e in events if e.event_type == 'attention_dip']
        if dips:
            recs.append(f"⚠️ Attention drop at :{dips[0].time_s:.0f}s — consider adding visual hook or cut.")

        # Identify engagement peak
        peaks = [e for e in events if 'peak_' in e.event_type]
        if peaks:
            peak = max(peaks, key=lambda e: e.magnitude)
            recs.append(f"✓ Peak engagement at :{peak.time_s:.0f}s — strengthen this moment (call-to-action, reveal, etc.).")

        # Memory encoding window
        mem_window = self.estimate_memory_encoding_window(construct_weights)
        if mem_window and mem_window.magnitude > 0.6:
            recs.append(f"🧠 Likely memory encoding from :{mem_window.window_s[0]:.0f}–:{mem_window.window_s[1]:.0f}s — logo/tagline placement optimal here.")

        # Overall arc
        start_eng = eng[0]
        end_eng = eng[-1]
        if end_eng > start_eng + 0.2:
            recs.append("✓ Strong closing — engagement rises toward CTA.")
        elif start_eng > end_eng + 0.2:
            recs.append("⚠️ Engagement fades — strengthen ending or move CTA earlier.")

        return recs if recs else ["No strong signal detected; recommend testing variations."]

    def plot_timeline(self, construct_weights: np.ndarray, output_path: str = 'timeline.png'):
        """
        Visualize engagement timeline + events.
        (Requires matplotlib; returns dict if plot not available.)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            t, eng = self.compute_engagement_timeline(construct_weights)
            events = self.detect_phase_transitions(construct_weights)

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(t, eng, linewidth=2.5, color='#2471a3', label=self.construct)
            ax.fill_between(t, eng, alpha=0.3, color='#2471a3')

            # Mark events
            for event in events:
                color = '#27ae60' if 'peak' in event.event_type else '#e74c3c' if 'dip' in event.event_type else '#f39c12'
                ax.axvline(event.time_s, color=color, linestyle='--', alpha=0.5, linewidth=1)
                ax.text(event.time_s, 0.95, event.event_type[:8], rotation=90, fontsize=8, alpha=0.7)

            ax.set_xlabel('Time (seconds)', fontsize=11)
            ax.set_ylabel('Engagement (0–1)', fontsize=11)
            ax.set_title(f'Brain Trajectory: {self.construct.title()} Over Stimulus Duration', fontsize=12)
            ax.set_ylim([-0.05, 1.05])
            ax.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(output_path, dpi=130)
            print(f"Timeline plot saved to {output_path}")

        except ImportError:
            print("Matplotlib not available; returning data dict instead of plot.")
            return self.timeline_heatmap(construct_weights, plot=False)

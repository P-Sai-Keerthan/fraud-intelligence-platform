import { useEffect, useState } from 'react'
import { getMetrics } from '../api'
import { bandText, shortHash } from '../severity'
import { Badge, Card, ErrorState, Icon, MetaRow, Skeleton } from './ui'

const fmt = (v, d = 3) => (v == null || Number.isNaN(Number(v)) ? '—' : Number(v).toFixed(d))

function Kpi({ label, value, unit, hint, mono = false, compact = false }) {
  return (
    <div className="card px-4 py-4 min-w-0">
      <div className="eyebrow">{label}</div>
      <div className="mt-2 flex items-baseline gap-1.5 min-w-0">
        <span className={`${mono ? 'mono text-[15px]' : compact ? 'text-[20px] tnum' : 'text-[28px] tnum'} leading-none truncate`} style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{value}</span>
        {unit && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{unit}</span>}
      </div>
      {hint && <div className="text-[11px] mt-2" style={{ color: 'var(--text-muted)' }}>{hint}</div>}
    </div>
  )
}

function ConfusionMatrix({ cm }) {
  if (!cm) return null
  const Cell = ({ n, label, tone }) => (
    <div className="panel px-3 py-3 text-center">
      <div className="mono text-lg tnum" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{Number(n).toLocaleString()}</div>
      <div className="flex items-center justify-center gap-1.5 mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
        <span className="rounded-full" style={{ width: 6, height: 6, background: tone }} />{label}
      </div>
    </div>
  )
  return (
    <div>
      <div className="grid gap-2 text-[10.5px] mono mb-1.5" style={{ gridTemplateColumns: '84px 1fr 1fr', color: 'var(--text-faint)' }}>
        <span /><span className="text-center">PREDICTED LEGITIMATE</span><span className="text-center">PREDICTED FRAUD</span>
      </div>
      <div className="grid gap-2 items-center" style={{ gridTemplateColumns: '84px 1fr 1fr' }}>
        <span className="mono text-[10.5px]" style={{ color: 'var(--text-faint)' }}>ACTUAL LEGIT.</span>
        <Cell n={cm.true_negative} label="True negative" tone="var(--risk-low)" />
        <Cell n={cm.false_positive} label="False positive" tone="var(--risk-medium)" />
        <span className="mono text-[10.5px]" style={{ color: 'var(--text-faint)' }}>ACTUAL FRAUD</span>
        <Cell n={cm.false_negative} label="False negative" tone="var(--risk-critical)" />
        <Cell n={cm.true_positive} label="True positive" tone="var(--risk-low)" />
      </div>
    </div>
  )
}

function Mini({ label, value }) {
  return (
    <div className="panel px-3 py-2.5">
      <div className="eyebrow" style={{ fontSize: 9.5 }}>{label}</div>
      <div className="mono text-base tnum mt-1" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{value}</div>
    </div>
  )
}

function ModelBlock({ title, subtitle, eyebrow, data, extra }) {
  if (!data) return null
  const ep = data.episodes
  return (
    <Card title={title} eyebrow={eyebrow} icon="cpu">
      <p className="text-xs -mt-1 mb-4" style={{ color: 'var(--text-muted)' }}>{subtitle}</p>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        <Mini label="Precision" value={fmt(data.precision)} />
        <Mini label="Recall" value={fmt(data.recall)} />
        <Mini label="F1 score" value={fmt(data.f1_score)} />
        <Mini label="False-positive rate" value={data.false_positive_rate != null ? `${(data.false_positive_rate * 100).toFixed(2)}%` : '—'} />
      </div>
      <ConfusionMatrix cm={data.confusion_matrix} />
      <dl className="mt-4">
        {data.test_set_size != null && (
          <MetaRow label="Test set">{Number(data.test_set_size).toLocaleString()} transactions{data.fraud_rate_pct != null ? ` · ${data.fraud_rate_pct}% fraud` : ''}</MetaRow>
        )}
        {data.threshold != null && <MetaRow label="Decision threshold (score 0–1)">{data.threshold}{data.threshold_source ? ` · ${data.threshold_source}` : ''}</MetaRow>}
        {ep && <MetaRow label="Fraud episodes detected">{ep.episodes_detected} / {ep.fraud_episodes}</MetaRow>}
        {ep?.first_fraud_recall != null && <MetaRow label="First fraud of an episode caught">{(ep.first_fraud_recall * 100).toFixed(1)}%</MetaRow>}
        {extra}
      </dl>
    </Card>
  )
}

const ci = (m, d = 3) => (m?.ci95 ? `95% CI ${Number(m.ci95[0]).toFixed(d)} – ${Number(m.ci95[1]).toFixed(d)}` : '')

function Gate({ ok, children }) {
  return (
    <li className="flex items-start gap-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
      <span className="mt-0.5 shrink-0" style={{ color: ok ? 'var(--risk-low)' : 'var(--risk-critical)' }}><Icon name={ok ? 'check' : 'x'} size={13} /></span>
      <span>{children}</span>
    </li>
  )
}

// Stage C final hold-out slice for the loaded artifact (GET /metrics -> final_holdout_evaluation).
// Every number is read from the backend's copy of final_holdout_report.json.
function FinalHoldout({ fh, info }) {
  const m = fh.primary?.metrics || {}
  const c = fh.primary?.counts || {}
  const g = fh.gates?.result || {}
  const nc = fh.new_customer || {}
  const blockers = info?.selection?.promotion_blockers || []
  const caught = c.true_positives != null ? `${Number(c.true_positives).toLocaleString()} / ${(Number(c.true_positives) + Number(c.false_negatives)).toLocaleString()}` : '—'
  return (
    <>
      <div className="eyebrow">Stage C final hold-out · customers with full history · frozen Policy B cut-off</div>
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3 -mt-2">
        <Kpi label="Recall" value={fmt(m.recall?.value)} hint={ci(m.recall)} />
        <Kpi label="Legit. alerts" value={m.legit_alerts_per_1000 ? Number(m.legit_alerts_per_1000.value).toFixed(2) : '—'} unit="/ 1,000" hint={ci(m.legit_alerts_per_1000, 2)} />
        <Kpi label="Fraud caught" value={caught} compact hint="Fraud transactions in the hold-out" />
        <Kpi label="PR-AUC" value={fmt(m.pr_auc?.value)} hint={ci(m.pr_auc)} />
        <Kpi label="ROC-AUC" value={fmt(m.roc_auc?.value)} hint={ci(m.roc_auc)} />
        <Kpi label="Production recall" value={fmt(fh.primary?.production_recall)} hint="Same hold-out, production model" />
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5 items-start">
        <Card title="Acceptance gates" eyebrow="Pre-registered · Stage C" icon="check"
          actions={<Badge tone={g.eligible ? 'good' : 'danger'}>{g.eligible ? 'All gates passed' : 'Gate failed'}</Badge>}>
          <ul className="space-y-2">
            <Gate ok={g.alert_budget_upper_bound_at_most_10}>Legitimate alerts: upper bound of the 95% interval at most 10 per 1,000</Gate>
            <Gate ok={g.recall_at_least_0_40}>Recall at least 0.40</Gate>
            <Gate ok={g.recall_lower_bound_above_production}>Lower bound of recall above the production model's recall</Gate>
          </ul>
          <dl className="mt-4">
            <MetaRow label="Hold-out transactions">{Number(fh.primary?.rows).toLocaleString()}</MetaRow>
            <MetaRow label="Fraud transactions · episodes">{Number(fh.primary?.fraud_transactions).toLocaleString()} · {Number(fh.primary?.fraud_episodes).toLocaleString()}</MetaRow>
            <MetaRow label="Outcome" mono={false}>{fh.outcome}</MetaRow>
            <MetaRow label="Promotion" mono={false}>Not promoted</MetaRow>
          </dl>
        </Card>
        <Card title="Known limitation: new customers" eyebrow="Promotion blocker" icon="alert"
          actions={<Badge tone="danger">Open</Badge>}>
          <div className="grid grid-cols-2 gap-2 mb-3">
            <Mini label="Legit. alerts / 1,000 (new customers)" value={nc.metrics?.legit_alerts_per_1000 ? Number(nc.metrics.legit_alerts_per_1000.value).toFixed(1) : '—'} />
            <Mini label="Legit. alerts / 1,000 (full history)" value={m.legit_alerts_per_1000 ? Number(m.legit_alerts_per_1000.value).toFixed(1) : '—'} />
          </div>
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{nc.limitation}</p>
          {blockers.length > 0 && (
            <>
              <div className="eyebrow mt-4 mb-2">Remaining before any promotion</div>
              <ul className="space-y-1.5">
                {blockers.map((b) => (
                  <li key={b} className="flex items-start gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
                    <span className="mt-1.5 rounded-full shrink-0" style={{ width: 5, height: 5, background: 'var(--risk-medium)' }} />{b}
                  </li>
                ))}
              </ul>
            </>
          )}
        </Card>
      </div>
    </>
  )
}


// ---- Step 4D: LSTM -> selected classifier (GET /metrics -> downstream_evaluation) ----
// Every number is copied by the backend from models/evaluation/downstream/*.json.

const pct = (v, d = 1) => (v == null ? '—' : `${(Number(v) * 100).toFixed(d)}%`)
const sd = (s, d = 3) => (s?.mean == null ? '—' : `${Number(s.mean).toFixed(d)}${s.sd != null ? ` ± ${Number(s.sd).toFixed(d)}` : ''}`)
const ciText = (c, d = 3) => (c ? `${Number(c[0]).toFixed(d)} – ${Number(c[1]).toFixed(d)}` : '—')

function ComparisonTable({ rows }) {
  const cols = [
    ['pr_auc', 'PR-AUC', 3], ['roc_auc', 'ROC-AUC', 3], ['precision', 'Precision', 3], ['recall', 'Recall', 3],
    ['f1', 'F1', 3], ['legit_alerts_per_1000', 'Legit alerts / 1,000', 2], ['first_fraud_recall', 'First fraud caught', 3],
    ['accuracy', 'Accuracy', 4],
  ]
  return (
    <div className="overflow-auto rounded-lg border" style={{ borderColor: 'var(--border)' }}>
      <table className="data-table">
        <thead>
          <tr>
            <th>Classifier after the LSTM</th>
            {cols.map(([k, label]) => <th key={k} className="num" style={{ textAlign: 'right' }}>{label}</th>)}
            <th className="num" style={{ textAlign: 'right' }}>Brier</th>
            <th className="num" style={{ textAlign: 'right' }}>ms / row</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.family}>
              <td style={{ color: 'var(--text-primary)', fontWeight: r.selected ? 600 : 400 }}>
                {r.label}{r.selected ? ' · selected' : r.family === 'dnn' ? ' · previous' : ''}
              </td>
              {cols.map(([k, , d]) => <td key={k} className="num mono">{sd(r[k], d)}</td>)}
              <td className="num mono">{r.brier_score != null ? Number(r.brier_score).toFixed(4) : '—'}</td>
              <td className="num mono">{r.single_row_ms != null ? Number(r.single_row_ms).toFixed(1) : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function HoldoutTable({ block }) {
  if (!block) return null
  const names = ['selected', 'dnn_seed14', 'production']
  const rows = [['pr_auc', 'PR-AUC', 3], ['roc_auc', 'ROC-AUC', 3], ['precision', 'Precision', 3], ['recall', 'Recall', 3],
    ['legit_alerts_per_1000', 'Legit alerts / 1,000', 2], ['first_fraud_recall', 'First fraud caught', 3],
    ['episode_detection_rate', 'Episodes detected', 3]]
  return (
    <div className="overflow-auto rounded-lg border" style={{ borderColor: 'var(--border)' }}>
      <table className="data-table">
        <thead>
          <tr>
            <th>Metric</th>
            {names.map((n) => <th key={n} className="num" style={{ textAlign: 'right' }}>{block.models[n]?.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, label, d]) => (
            <tr key={k}>
              <td>{label}</td>
              {names.map((n) => {
                const m = block.models[n]?.metrics?.[k]
                return (
                  <td key={n} className="num mono" title={m?.ci95 ? `95% CI ${ciText(m.ci95, d)}` : ''}>
                    {m?.value != null ? Number(m.value).toFixed(d) : '—'}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Reliability({ bins }) {
  if (!bins?.length) return null
  return (
    <div className="overflow-auto rounded-lg border" style={{ borderColor: 'var(--border)' }}>
      <table className="data-table">
        <thead>
          <tr>
            <th>Fraud Score band</th>
            <th className="num" style={{ textAlign: 'right' }}>Transactions</th>
            <th className="num" style={{ textAlign: 'right' }}>Mean score</th>
            <th className="num" style={{ textAlign: 'right' }}>Observed fraud rate</th>
          </tr>
        </thead>
        <tbody>
          {bins.map((b) => (
            <tr key={b.bin[0]}>
              <td className="mono">{(b.bin[0] * 100).toFixed(0)}–{(b.bin[1] * 100).toFixed(0)}</td>
              <td className="num mono">{Number(b.rows).toLocaleString()}</td>
              <td className="num mono">{pct(b.mean_score)}</td>
              <td className="num mono">{pct(b.observed_fraud_rate)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function DownstreamSelection({ de }) {
  const fh = de.final_holdout || {}
  const p = fh.primary || {}
  const sel = p.models?.selected?.metrics || {}
  const dev = de.development || {}
  const g = fh.gates || {}
  const diff = p.paired_differences?.['selected - dnn_seed14'] || {}
  const selectedRow = (dev.comparison || []).find((r) => r.selected)
  const early = fh.new_customer_early_history?.models || {}
  return (
    <>
      <div className="eyebrow">Fresh hold-out (seeds 501–505, scored once) · customers with at least 10 earlier transactions</div>
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3 -mt-2">
        <Kpi label="PR-AUC" value={fmt(sel.pr_auc?.value)} hint={ci(sel.pr_auc)} />
        <Kpi label="Recall" value={fmt(sel.recall?.value)} hint={ci(sel.recall)} />
        <Kpi label="Precision" value={fmt(sel.precision?.value)} hint={ci(sel.precision)} />
        <Kpi label="Legit. alerts" value={sel.legit_alerts_per_1000 ? Number(sel.legit_alerts_per_1000.value).toFixed(2) : '—'} unit="/ 1,000" hint={ci(sel.legit_alerts_per_1000, 2)} />
        <Kpi label="First fraud caught" value={fmt(sel.first_fraud_recall?.value)} hint={ci(sel.first_fraud_recall)} />
        <Kpi label="ROC-AUC" value={fmt(sel.roc_auc?.value)} hint={ci(sel.roc_auc)} />
      </div>

      <Card title="Classifier comparison" eyebrow={`Development data · ${dev.datasets?.length || 5} datasets · mean ± sd over ${dev.training_seeds?.length || 5} training seeds`} icon="cpu"
        actions={<Badge tone="good">{selectedRow ? `${selectedRow.label} selected` : 'Selected'}</Badge>}>
        <p className="text-xs -mt-1 mb-3" style={{ color: 'var(--text-muted)' }}>
          Every classifier receives the same inputs: the 9 behavioral features and the Risk Score of the same LSTM. The winner
          was fixed by a rule written before scoring: highest mean PR-AUC among candidates that clearly beat the DNN. Accuracy is
          shown only to make the point that it barely differs (about 99% for every model, because 99.5% of transactions are legitimate).
        </p>
        <ComparisonTable rows={dev.comparison || []} />
        <p className="text-[11px] mt-2" style={{ color: 'var(--text-muted)' }}>
          {Number(dev.population?.rows || 0).toLocaleString()} transactions, {Number(dev.population?.fraud_transactions || 0).toLocaleString()} fraud.
          Each model at its own cut-off chosen on the validation period (false-positive rate within 1%). {dev.decision?.note ? `Decision: ${dev.decision.note}.` : ''}
        </p>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5 items-start">
        <Card title="Fresh hold-out confirmation" eyebrow="Pre-registered · scored once" icon="check"
          actions={<Badge tone={fh.confirmed ? 'good' : 'danger'}>{fh.confirmed ? 'Confirmed' : 'Not confirmed'}</Badge>}>
          <HoldoutTable block={p} />
          <ul className="space-y-2 mt-4">
            <Gate ok={g.C1_pr_auc_higher_than_seed14_dnn?.holds}>PR-AUC above the DNN with the same LSTM: difference {fmt(diff.pr_auc?.difference)} (95% CI {ciText(diff.pr_auc?.ci95)})</Gate>
            <Gate ok={g['C2_recall_at_least_0.40']?.holds}>Recall at least 0.40</Gate>
            <Gate ok={g.C3_alert_burden_not_materially_higher?.holds}>Legitimate alerts not more than 1 per 1,000 above the DNN: difference {diff.legit_alerts_per_1000?.difference != null ? Number(diff.legit_alerts_per_1000.difference).toFixed(2) : '—'} (95% CI {ciText(diff.legit_alerts_per_1000?.ci95, 2)})</Gate>
          </ul>
          <dl className="mt-4">
            <MetaRow label="Hold-out transactions">{Number(p.rows || 0).toLocaleString()}</MetaRow>
            <MetaRow label="Fraud transactions · episodes">{Number(p.fraud_transactions || 0).toLocaleString()} · {Number(p.fraud_episodes || 0).toLocaleString()}</MetaRow>
            <MetaRow label="Outcome" mono={false}>{fh.outcome}</MetaRow>
          </dl>
          <div className="mt-4"><ConfusionMatrix cm={p.models?.selected?.confusion_matrix} /></div>
        </Card>

        <div className="space-y-5">
          <Card title="Calibration" eyebrow="Fresh hold-out · selected model" icon="info"
            actions={<Badge tone="warn">Not a calibrated probability</Badge>}>
            <p className="text-xs -mt-1 mb-3" style={{ color: 'var(--text-muted)' }}>
              If the Fraud Score were a probability, each band&apos;s observed fraud rate would match its mean score. Almost every
              transaction scores below 10, where the two agree; above that the observed rate is clearly higher than the score, so the
              Fraud Score is a ranking score, not a probability.
            </p>
            <Reliability bins={fh.selected_reliability} />
          </Card>
          <Card title="Known limitation: new customers" eyebrow="Fewer than 10 earlier transactions" icon="alert"
            actions={<Badge tone="danger">Open</Badge>}>
            <div className="grid grid-cols-2 gap-2 mb-3">
              <Mini label="Legit. alerts / 1,000 (selected)" value={early.selected?.metrics?.legit_alerts_per_1000 ? Number(early.selected.metrics.legit_alerts_per_1000.value).toFixed(1) : '—'} />
              <Mini label="Legit. alerts / 1,000 (DNN)" value={early.dnn_seed14?.metrics?.legit_alerts_per_1000 ? Number(early.dnn_seed14.metrics.legit_alerts_per_1000.value).toFixed(1) : '—'} />
              <Mini label="Recall (selected)" value={fmt(early.selected?.metrics?.recall?.value)} />
              <Mini label="Recall (DNN)" value={fmt(early.dnn_seed14?.metrics?.recall?.value)} />
            </div>
            <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              Without 10 earlier transactions the LSTM is not run. These figures come from the fresh new-customer hold-out (seeds 511–515).
            </p>
          </Card>
        </div>
      </div>
      <p className="text-[11px]" style={{ color: 'var(--text-faint)' }}>
        {de.scores_note} Full method and every number: {de.sources?.report}.
      </p>
    </>
  )
}

// A compact technical card built from GET /model-info.
function ModelCard({ info }) {
  if (!info) {
    return (
      <Card title="Model information" eyebrow="Observability" icon="cpu">
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Model information is not available (GET /model-info failed).</p>
      </Card>
    )
  }
  const calibrated = info.score_semantics?.calibrated_probabilities
  const bands = info.thresholds?.alert_bands || {}
  const fv = info.features?.training_feature_version
  const frozen = info.thresholds?.frozen_cutoffs
  return (
    <Card title="Model information" eyebrow="Observability" icon="cpu"
      actions={<Badge tone={info.status === 'PRODUCTION' ? 'good' : 'warn'}>{info.status === 'PRODUCTION' ? 'PRODUCTION' : info.status?.startsWith('PREVIOUS') ? 'PREVIOUS DEFAULT' : 'NOT DEPLOYED'}</Badge>}>
      <dl>
        <MetaRow label="Model set">{info.model_set}</MetaRow>
        <MetaRow label="Model version">{info.model_version}</MetaRow>
        <MetaRow label="Architecture" mono={false}>{info.architecture ? `${info.architecture} · ` : ''}{info.uses_lstm ? `LSTM risk score → ${info.downstream_classifier?.label || 'DNN'} classifier` : 'DNN classifier (no sequence model)'}</MetaRow>
        {info.downstream_classifier?.params && <MetaRow label="Classifier settings">{Object.entries(info.downstream_classifier.params).map(([k, v]) => `${k}=${v === null ? 'none' : v}`).join(', ')}</MetaRow>}
        {info.training_seed != null && <MetaRow label="Training seed">{info.training_seed}</MetaRow>}
        {info.status && <MetaRow label="Status">{info.status}</MetaRow>}
        {info.selection?.artifact && <MetaRow label="Selected artifact">{info.selection.artifact}</MetaRow>}
        <MetaRow label="Dataset version">{info.dataset?.version || '—'}{info.dataset?.sha256 ? ` · sha256 ${shortHash(info.dataset.sha256)}` : ''}</MetaRow>
        <MetaRow label="Feature version">
          {fv?.name || '—'} · {info.features?.columns?.length ?? '—'} features{info.features?.list_sha256 ? ` · ${shortHash(info.features.list_sha256)}` : ''}
        </MetaRow>
        <MetaRow label="Score semantics" mono={false}>
          {calibrated === false ? 'Model scores, not calibrated probabilities' : calibrated === true ? 'Calibrated probabilities' : '—'}
        </MetaRow>
        <MetaRow label="Minimum history for LSTM">{info.cold_start?.min_prior_transactions != null ? `${info.cold_start.min_prior_transactions} earlier transactions` : '—'}</MetaRow>
      </dl>

      <div className="mt-4">
        <div className="flex items-center justify-between mb-2">
          <span className="eyebrow">Alert bands</span>
          <Badge tone="warn" title={info.thresholds?.alert_bands_status || ''}>Threshold status: legacy fixed bands</Badge>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(bands).map(([level, rule]) => (
            <div key={level} className="panel px-3 py-2">
              <div className="text-xs" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>{level}</div>
              <div className="mono text-[11px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{bandText(rule)}</div>
            </div>
          ))}
        </div>
        {info.thresholds?.alert_bands_status && (
          <p className="text-[11px] mt-2" style={{ color: 'var(--text-muted)' }}>{info.thresholds.alert_bands_status}.</p>
        )}
      </div>

      {frozen && (
        <div className="mt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="eyebrow">Evaluation cut-offs (validation period)</span>
            <Badge tone="warn" title={frozen.status || ''}>Not applied to live scoring</Badge>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div className="panel px-3 py-2">
              <div className="text-xs" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Policy B (alert)</div>
              <div className="mono text-[11px] mt-0.5" style={{ color: 'var(--text-muted)' }}>fraud score ≥ {(frozen.policy_b * 100).toFixed(2)}</div>
            </div>
            <div className="panel px-3 py-2">
              <div className="text-xs" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>Critical</div>
              <div className="mono text-[11px] mt-0.5" style={{ color: 'var(--text-muted)' }}>fraud score ≥ {(frozen.critical * 100).toFixed(2)}</div>
            </div>
          </div>
          <p className="text-[11px] mt-2" style={{ color: 'var(--text-muted)' }}>{frozen.source}. {frozen.status}.</p>
        </div>
      )}

      {info.score_semantics && (
        <details className="mt-4">
          <summary className="text-xs cursor-pointer" style={{ color: 'var(--text-secondary)' }}>What the scores mean</summary>
          <p className="text-[11px] mt-2" style={{ color: 'var(--text-muted)' }}><span className="mono" style={{ color: 'var(--text-secondary)' }}>fraud score</span> — {String(info.score_semantics.fraud_probability || '')}</p>
          <p className="text-[11px] mt-1.5" style={{ color: 'var(--text-muted)' }}><span className="mono" style={{ color: 'var(--text-secondary)' }}>risk score</span> — {String(info.score_semantics.risk_score || '')}</p>
        </details>
      )}
    </Card>
  )
}

export default function ModelPerformance({ modelInfo }) {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const load = () => {
    setLoading(true)
    setError('')
    getMetrics()
      .then(setMetrics)
      .catch(() => setError('Could not load model metrics. Is the backend running?'))
      .finally(() => setLoading(false))
  }
  useEffect(load, [])

  if (loading) {
    return (
      <div className="space-y-5">
        <Skeleton style={{ height: 76 }} />
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">{[0, 1, 2, 3, 4, 5].map((i) => <Skeleton key={i} style={{ height: 96 }} />)}</div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5"><Skeleton style={{ height: 380 }} /><Skeleton style={{ height: 380 }} /></div>
      </div>
    )
  }
  if (error) {
    return <ErrorState onRetry={load}>{error}</ErrorState>
  }

  const info = modelInfo || metrics?.model_metadata || null
  const modelSet = metrics?.model_set
  const de = metrics?.downstream_evaluation?.available ? metrics.downstream_evaluation : null
  if (de) {
    return (
      <div className="space-y-5">
        <div className="card px-5 py-4" style={{ borderColor: 'rgba(47,191,143,0.3)' }}>
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="good">PRODUCTION MODEL</Badge>
            <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>{modelSet}{metrics?.model_version ? ` · ${metrics.model_version}` : ''}</span>
          </div>
          <p className="text-xs mt-2 max-w-4xl" style={{ color: 'var(--text-muted)' }}>
            {info?.architecture || 'LSTM + classifier'}: the LSTM turns the customer&apos;s previous 10 transactions into a temporal Risk Score,
            and the {info?.downstream_classifier?.label || 'classifier'} combines it with the 9 behavioral features into the Fraud Score.
            The classifier was chosen among four candidates on development data by a pre-registered rule and confirmed once on fresh
            data. All data is synthetic; these figures do not describe real banking traffic.
          </p>
        </div>
        <DownstreamSelection de={de} />
        <ModelCard info={info} />
      </div>
    )
  }
  const isCandidate = modelSet && modelSet !== 'production'
  const ev = metrics?.candidate_evaluation
  const fh = isCandidate && metrics?.final_holdout_evaluation?.available ? metrics.final_holdout_evaluation : null
  const main = isCandidate ? (ev?.available ? ev.overall_test : null) : metrics?.dnn_fraud_classifier
  const cm = main?.confusion_matrix
  const caught = cm ? `${Number(cm.true_positive).toLocaleString()} / ${(Number(cm.true_positive) + Number(cm.false_negative)).toLocaleString()}` : '—'

  return (
    <div className="space-y-5">
      <div className="card px-5 py-4 flex flex-wrap items-center justify-between gap-4"
        style={{ borderColor: isCandidate ? 'rgba(242,180,31,0.35)' : 'rgba(47,191,143,0.3)' }}>
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            {isCandidate ? (
              <>
                <Badge tone="warn">EVALUATION CANDIDATE</Badge>
                <Badge tone="danger" title="This model set is loaded for evaluation in this session. Production remains the default model set.">NOT DEPLOYED</Badge>
              </>
            ) : (
              <Badge tone={info?.status === 'PRODUCTION' ? 'good' : 'warn'}>{info?.status === 'PRODUCTION' ? 'PRODUCTION MODEL' : 'PREVIOUS DEFAULT (v1)'}</Badge>
            )}
            <span className="mono text-xs" style={{ color: 'var(--text-secondary)' }}>{modelSet || 'model set unknown'}{metrics?.model_version ? ` · ${metrics.model_version}` : ''}</span>
          </div>
          <p className="text-xs mt-2 max-w-4xl" style={{ color: 'var(--text-muted)' }}>
            {fh
              ? 'Validated candidate, not yet deployed. The figures below are the Stage C final hold-out result for exactly these weights, at the frozen Policy B cut-off, on synthetic v2 data. Live scoring in this session still uses the fixed alert bands, and production remains the default model set.'
              : isCandidate
              ? (ev?.available
                ? 'Evaluation of this candidate on the v2 synthetic test period, at its validation-chosen F1 threshold. That threshold is not applied by live scoring, which uses the fixed alert bands. Loaded here for evaluation only; production remains the default model set.'
                : `No candidate evaluation is available (${ev?.reason || 'unknown reason'}). The production evaluation does not describe this model set, so it is not shown.`)
              : 'Time-based evaluation on dataset v1: evaluation copies of the production models are trained on the earliest transactions, the decision threshold is chosen on the following period, and the figures below are measured on the latest period.'}
          </p>
          {!isCandidate && metrics?.evaluation?.evaluates && (
            <p className="text-[11px] mt-1.5 max-w-4xl" style={{ color: 'var(--text-faint)' }}>
              Evaluated: {metrics.evaluation.evaluates}. Scores are not calibrated probabilities. Full report: GET /metrics/report.
            </p>
          )}
        </div>
      </div>

      {fh && <FinalHoldout fh={fh} info={info} />}

      {main && (
        <>
          <div className="eyebrow">{isCandidate ? 'Candidate · v2 test period' : 'Real-time detector · DNN fraud classifier'}</div>
          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3 -mt-2">
            <Kpi label="PR-AUC" value={fmt(main.pr_auc)} hint="Area under precision–recall" />
            <Kpi label="ROC-AUC" value={fmt(main.auc_roc)} hint="Ranking quality" />
            <Kpi label="Recall" value={fmt(main.recall)} hint="Share of fraud flagged" />
            <Kpi label="Alert rate" value={main.alerts_per_1000 != null ? Number(main.alerts_per_1000).toFixed(2) : '—'} unit="/ 1,000" hint="Alerts per 1,000 transactions" />
            <Kpi label="Fraud caught" value={caught} hint="Fraud transactions in the test set" />
            <Kpi label="Model version" value={metrics?.model_version || '—'} mono hint={modelSet ? `model set ${modelSet}` : ''} />
          </div>
        </>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5 items-start">
        {isCandidate ? (
          ev?.available && (
            <ModelBlock
              eyebrow="Evaluation candidate"
              title={info?.uses_lstm ? 'DNN Fraud Classifier (LSTM risk score → DNN)' : 'DNN Fraud Classifier (DNN only)'}
              subtitle={`v2 test period: ${ev.rows?.test?.toLocaleString()} transactions, ${ev.rows?.test_fraud_episodes} fraud episodes. First fraud caught in ${ev.episodes_test?.first_fraud_detected} of ${ev.episodes_test?.first_fraud_transactions} episodes.`}
              data={ev.overall_test}
            />
          )
        ) : (
          <>
            <ModelBlock
              eyebrow="Production model · stage 2"
              title="DNN Fraud Classifier"
              subtitle="Classifies each transaction from its own features plus the LSTM risk score. This is the real-time detection signal."
              data={metrics?.dnn_fraud_classifier}
            />
            <ModelBlock
              eyebrow="Production model · stage 1"
              title="LSTM Temporal Risk Model"
              subtitle="Reads the customer's previous 10 transactions and produces the temporal risk signal used by the DNN. It is trained to score whether the next transaction is fraudulent; it does not detect the first fraud of an episode (see below)."
              data={metrics?.lstm_risk_predictor}
              extra={<>
                <MetaRow label="PR-AUC · ROC-AUC">{fmt(metrics?.lstm_risk_predictor?.pr_auc)} · {fmt(metrics?.lstm_risk_predictor?.auc_roc)}</MetaRow>
              </>}
            />
          </>
        )}
        <div className={isCandidate && ev?.available ? '' : 'xl:col-span-2'}>
          <ModelCard info={info} />
        </div>
      </div>
    </div>
  )
}

/**
 * Screening evaluation metrics for the dashboard matrix.
 * Sourced from frozen held-out test lots (L08/L10/L15/L16, 961 parts, 32 defective).
 * Numbers match reports/metrics.json (improve-recall) and metrics_v1.json — not inflated demos.
 */
import metrics from './metrics.json'
import metricsV1 from './metrics_v1.json'

function pct(rate, digits = 1) {
  if (rate == null || !Number.isFinite(Number(rate))) return '—'
  return `${(Number(rate) * 100).toFixed(digits)}%`
}

function num(v, digits = 3) {
  if (v == null || !Number.isFinite(Number(v))) return '—'
  return Number(v).toFixed(digits)
}

function rowFromBlock(label, block, highlight = false) {
  if (!block) return null
  const cm = block.confusion_matrix || {}
  return {
    system: label,
    highlight,
    recall: block.recall,
    recallLabel: pct(block.recall),
    precision: block.precision,
    precisionLabel: pct(block.precision),
    f2: block.f2,
    f2Label: num(block.f2, 3),
    overkill: block.overkill_rate,
    overkillLabel: pct(block.overkill_rate),
    escapes: block.escapes,
    falseRejects: block.false_rejects,
    cost: block.cost,
    caught: cm.tp != null && block.n_defective != null ? `${cm.tp}/${block.n_defective}` : null,
  }
}

function maeRow(name, block, isDefault = false) {
  if (!block) return null
  const pp = block.per_param || {}
  return {
    model: isDefault ? `${name} (default)` : name,
    highlight: isDefault,
    overallMae: block.overall_mae,
    overallMaeLabel: num(block.overall_mae, 4),
    nmaePct: block.overall_nmae_pct,
    nmaeLabel: `${num(block.overall_nmae_pct, 2)}%`,
    iddq: pp.Iddq?.mae,
    iddqLabel: num(pp.Iddq?.mae, 4),
    leakage: pp.Leakage?.mae,
    leakageLabel: num(pp.Leakage?.mae, 4),
    propDelay: pp.Prop_Delay?.mae,
    propDelayLabel: num(pp.Prop_Delay?.mae, 4),
  }
}

const current = metrics
const baseline = metricsV1
const reject = current.combined_reject_only || {}
const rejectReview = current.combined_reject_or_review || {}
const moduleA = current.module_a || {}
const moduleBEarly = current.module_b_early_24h || {}
const rules = current.rules_only || {}
const mae = current.mae_value_168h || {}
const defaultB = current.module_b_default || 'ridge'
const cm = reject.confusion_matrix || {}

export const screeningMetrics = {
  source: 'improve-recall evaluation (frozen test lots)',
  baselineSource: 'v1.0 frozen baseline',
  eval: {
    n: reject.n ?? 961,
    nDefective: reject.n_defective ?? 32,
    testLots: current.test_lots || ['L08', 'L10', 'L15', 'L16'],
    costFn: current.thresholds?.cost_fn ?? 50,
    costFp: current.thresholds?.cost_fp ?? 1,
  },
  highlights: [
    {
      key: 'reject_recall',
      label: 'REJECT recall',
      value: pct(reject.recall),
      detail: cm.tp != null ? `${cm.tp}/${reject.n_defective} defects` : null,
      note: 'Combined hard reject',
    },
    {
      key: 'reject_review_recall',
      label: 'REJECT+REVIEW recall',
      value: pct(rejectReview.recall),
      detail:
        rejectReview.confusion_matrix?.tp != null
          ? `${rejectReview.confusion_matrix.tp}/${rejectReview.n_defective}`
          : null,
      note: 'Hold band included',
    },
    {
      key: 'overkill',
      label: 'REJECT overkill',
      value: pct(reject.overkill_rate),
      detail: reject.false_rejects != null ? `${reject.false_rejects} false rejects` : null,
      note: 'Good parts flagged',
    },
    {
      key: 'mae',
      label: `Module B MAE (${defaultB})`,
      value: num(mae[defaultB]?.overall_mae, 4),
      detail: mae[defaultB]?.overall_nmae_pct != null
        ? `nMAE ${num(mae[defaultB].overall_nmae_pct, 2)}%`
        : null,
      note: 'Value_168h forecast',
    },
  ],
  detectionRows: [
    rowFromBlock('Rules only', rules),
    rowFromBlock('Module A', moduleA),
    rowFromBlock('Module B @24h', moduleBEarly),
    rowFromBlock('Combined REJECT', reject, true),
    rowFromBlock('Combined REJECT+REVIEW', rejectReview, true),
  ].filter(Boolean),
  maeRows: ['physics', 'hybrid', 'xgboost', 'ridge', 'naive']
    .map((name) => maeRow(name, mae[name], name === defaultB))
    .filter(Boolean),
  moduleSummary: [
    {
      module: 'Module A',
      role: 'Lot-relative anomaly',
      recall: pct(moduleA.recall),
      overkill: pct(moduleA.overkill_rate),
      escapes: moduleA.escapes,
    },
    {
      module: 'Module B',
      role: `Early drift · ${defaultB} forecast`,
      recall: pct(moduleBEarly.recall),
      overkill: pct(moduleBEarly.overkill_rate),
      mae: num(mae[defaultB]?.overall_mae, 4),
    },
    {
      module: 'Combined',
      role: 'Rules + A + B decide',
      recall: pct(reject.recall),
      overkill: pct(reject.overkill_rate),
      rejectReviewRecall: pct(rejectReview.recall),
    },
  ],
  baselineCompare: {
    rejectRecallV1: pct(baseline.combined_reject_only?.recall),
    rejectRecallV2: pct(reject.recall),
    rejectReviewV1: pct(baseline.combined_reject_or_review?.recall),
    rejectReviewV2: pct(rejectReview.recall),
    maeRidgeV1: num(baseline.mae_value_168h?.ridge?.overall_mae, 4),
    maeRidgeV2: num(mae.ridge?.overall_mae, 4),
  },
}

export default screeningMetrics

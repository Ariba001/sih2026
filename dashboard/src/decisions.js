/** Screening decision copy for BurnTestr operators. */

export const DECISION_GUIDE = {
  ACCEPT: {
    key: 'ACCEPT',
    label: 'Accept',
    short: 'Cleared for use',
    meaning:
      'The component stays inside lot-relative anomaly and early-drift thresholds. Static datasheet and delta limits also pass.',
    action:
      'Release with the lot. No hold is required; keep the scores in the quality record for traceability.',
  },
  REVIEW: {
    key: 'REVIEW',
    label: 'Review',
    short: 'Engineering hold',
    meaning:
      'Scores sit in the gray band below hard reject but above the accept floor — a borderline lot-outlier or drift signal.',
    action:
      'Hold the part for engineering review. Compare Module A/B drivers, re-measure if needed, then accept or reject with a signed disposition.',
  },
  REJECT: {
    key: 'REJECT',
    label: 'Reject',
    short: 'Do not ship',
    meaning:
      'A hard rule failed (datasheet/delta) or the combined anomaly/drift score crossed the reject threshold. These calls are not waivable by default.',
    action:
      'Quarantine from the flight lot. Route to MRB only under formal process; do not mix with accepted inventory.',
  },
}

export function normalizeDecision(value) {
  const d = String(value || '').trim().toUpperCase()
  if (d === 'ACCEPTED') return 'ACCEPT'
  if (d === 'REJECTED') return 'REJECT'
  if (d === 'ACCEPT' || d === 'REVIEW' || d === 'REJECT') return d
  return d || 'ACCEPT'
}

export function decisionMeta(value) {
  const key = normalizeDecision(value)
  return DECISION_GUIDE[key] || {
    key,
    label: key,
    short: key,
    meaning: 'Screening decision from BurnTestr.',
    action: 'Follow your site quality procedure for this disposition.',
  }
}

function cleanText(value) {
  const s = String(value ?? '').trim()
  if (!s || s.toLowerCase() === 'nan' || s.toLowerCase() === 'none') return ''
  return s
}

function formatCodes(codes) {
  const raw = cleanText(codes)
  if (!raw) return ''
  return raw
    .split(/[;,]/)
    .map((c) => c.trim())
    .filter(Boolean)
    .join(', ')
}

/**
 * Prefer API `description`; otherwise build clear operator copy.
 * Long pipeline `explanation` is not used as the primary table text.
 */
export function describeRow(row = {}) {
  const fromApi = cleanText(row.description)
  if (fromApi) return fromApi

  const decision = normalizeDecision(row.decision)
  const a = Number(row.score_a)
  const b = Number(row.score_b)
  const aOk = Number.isFinite(a)
  const bOk = Number.isFinite(b)
  const codes = formatCodes(row.reason_codes)
  const meta = decisionMeta(decision)

  let detail = ''
  if (decision === 'ACCEPT') {
    detail = aOk && bOk
      ? `Within lot-relative and drift thresholds (Module A ${a.toFixed(2)}, Module B ${b.toFixed(2)}). Ship-ready for the screened application; no operator hold required.`
      : meta.meaning
  } else if (decision === 'REVIEW') {
    detail = aOk && bOk
      ? `Near the decision boundary (Module A ${a.toFixed(2)}, Module B ${b.toFixed(2)}). ${meta.action}`
      : `${meta.meaning} ${meta.action}`
  } else if (decision === 'REJECT') {
    if (aOk && a >= 1e5) {
      detail =
        'Hard rule failure (datasheet limit or delta limit). Quarantine — do not waive without formal MRB.'
    } else {
      detail = aOk && bOk
        ? `Above reject threshold (Module A ${a.toFixed(2)}, Module B ${b.toFixed(2)}). ${meta.action}`
        : `${meta.meaning} ${meta.action}`
    }
  } else {
    detail = meta.meaning
  }

  if (codes) detail += ` Triggers: ${codes}.`
  return detail
}

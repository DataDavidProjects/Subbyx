import { ResultCardProps } from "./ResultCard.types"

export function ResultCard({
  decision,
  score,
  segment,
  reason,
  ruleTriggered,
  segmentReason,
  productionScore,
  shadowScore,
  scoredBy,
  isFraudTruth,
}: ResultCardProps) {
  const isBlock = decision === "BLOCK"
  const isApprove = decision === "APPROVE"

  const decisionColor = isBlock
    ? "border-red-300 bg-red-50"
    : isApprove
    ? "border-green-300 bg-green-50"
    : "border-yellow-300 bg-yellow-50"

  const badgeColor = isBlock
    ? "bg-red-600 text-white"
    : isApprove
    ? "bg-green-600 text-white"
    : "bg-yellow-500 text-white"

  return (
    <div className={`rounded-lg border-2 ${decisionColor} divide-y divide-gray-200`}>

      {/* ---- Decision + Score ---- */}
      <div className="flex items-center justify-between p-6">
        <div>
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
            Decision
          </p>
          <div className="flex items-center gap-2">
            <span className={`px-4 py-1.5 rounded-full text-sm font-bold ${badgeColor}`}>
              {decision}
            </span>
            {scoredBy === "shadow" && (
              <span className="px-2 py-0.5 bg-amber-100 text-amber-800 rounded text-xs font-semibold uppercase">
                Canary (Shadow)
              </span>
            )}
          </div>
        </div>

        <div className="text-right">
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
            Ground Truth
          </p>
          <span className={`px-2 py-0.5 rounded text-xs font-bold ${isFraudTruth ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"}`}>
            {isFraudTruth ? "FRAUD" : "CLEAN"}
          </span>
        </div>

        <div className="text-right">
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
            Score
          </p>
          <span className="text-2xl font-mono font-bold text-gray-900">
            {score !== null ? score.toFixed(4) : "—"}
          </span>
        </div>
      </div>

      {/* ---- Model Scores ---- */}
      {(productionScore != null || shadowScore != null) && (
        <div className="px-6 py-4">
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">
            Model Scores
          </p>
          <div className="grid grid-cols-2 gap-4">
            <ScoreCell
              label="Production"
              value={productionScore}
              active={scoredBy === "production"}
            />
            <ScoreCell
              label="Shadow (Canary)"
              value={shadowScore}
              active={scoredBy === "shadow"}
            />
          </div>
        </div>
      )}

      {/* ---- Segment ---- */}
      {segment && (
        <div className="px-6 py-4">
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
            Segment
          </p>
          <p className="font-semibold text-gray-900">{segment}</p>
          {segmentReason && (
            <p className="text-sm text-gray-600 mt-0.5">{segmentReason}</p>
          )}
        </div>
      )}

      {/* ---- Reason + Rule ---- */}
      <div className="px-6 py-4">
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
          Reason
        </p>
        <p className="text-gray-800">{reason}</p>

        {ruleTriggered && (
          <div className="mt-3">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-red-100 text-red-800 rounded text-xs font-semibold">
              Rule triggered: {ruleTriggered}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

function ScoreCell({
  label,
  value,
  active,
}: {
  label: string
  value?: number | null
  active?: boolean
}) {
  if (value == null) {
    return (
      <div className="text-center">
        <p className="text-xs text-gray-400">{label}</p>
        <p className="font-mono text-sm text-gray-300 mt-0.5">—</p>
      </div>
    )
  }

  return (
    <div className={`text-center rounded-md px-2 py-1.5 ${active ? "bg-white/60 ring-1 ring-gray-300" : ""}`}>
      <p className={`text-xs ${active ? "font-semibold text-gray-700" : "text-gray-500"}`}>
        {label}{active ? " (active)" : ""}
      </p>
      <p className="font-mono text-sm font-semibold text-gray-900 mt-0.5">
        {value.toFixed(4)}
      </p>
    </div>
  )
}

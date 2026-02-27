import { ResultCardProps } from "./ResultCard.types"

export function ResultCard({
  decision,
  score,
  segment,
  reason,
  ruleTriggered,
  segmentReason,
  shadowScore,
  canaryScore,
  scoredBy,
}: ResultCardProps) {
  const isBlock = decision === "BLOCK"
  const isApprove = decision === "APPROVE"

  return (
    <div
      className={`rounded-lg border-2 p-6 ${
        isBlock
          ? "border-red-300 bg-red-50"
          : isApprove
          ? "border-green-300 bg-green-50"
          : "border-yellow-300 bg-yellow-50"
      }`}
    >
      <div className="flex items-center justify-between mb-4">
        <span className="text-lg font-semibold">Decision:</span>
        <div className="flex items-center gap-2">
          {scoredBy === "canary" && (
            <span className="px-2 py-1 bg-amber-100 text-amber-800 rounded text-xs font-semibold uppercase">
              Canary
            </span>
          )}
          <span
            className={`px-4 py-2 rounded-full text-lg font-bold ${
              isBlock
                ? "bg-red-600 text-white"
                : isApprove
                ? "bg-green-600 text-white"
                : "bg-yellow-500 text-white"
            }`}
          >
            {decision}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-600">Score:</span>
          <span className="ml-2 font-mono font-semibold">
            {score !== null ? score.toFixed(4) : "N/A"}
          </span>
        </div>
        <div>
          <span className="text-gray-600">Segment:</span>
          <span className="ml-2 font-semibold">{segment || "N/A"}</span>
        </div>
        {scoredBy && (
          <div>
            <span className="text-gray-600">Scored by:</span>
            <span className={`ml-2 font-semibold ${
              scoredBy === "canary" ? "text-amber-700" : "text-gray-900"
            }`}>
              {scoredBy}
            </span>
          </div>
        )}
        {shadowScore !== null && shadowScore !== undefined && (
          <div>
            <span className="text-gray-600">Shadow Score:</span>
            <span className="ml-2 font-mono font-semibold">
              {shadowScore.toFixed(4)}
            </span>
          </div>
        )}
        {canaryScore !== null && canaryScore !== undefined && (
          <div>
            <span className="text-gray-600">Canary Score:</span>
            <span className="ml-2 font-mono font-semibold">
              {canaryScore.toFixed(4)}
            </span>
          </div>
        )}
      </div>

      <div className="mt-4 pt-4 border-t border-gray-200">
        <span className="text-gray-600 text-sm">Reason:</span>
        <p className="mt-1 text-gray-800">{reason}</p>
      </div>

      {segmentReason && (
        <div className="mt-2">
          <span className="text-gray-600 text-sm">Segment Reason:</span>
          <p className="mt-1 text-gray-800">{segmentReason}</p>
        </div>
      )}

      {ruleTriggered && (
        <div className="mt-2">
          <span className="text-gray-600 text-sm">Rule Triggered:</span>
          <span className="ml-2 px-2 py-1 bg-red-100 text-red-800 rounded text-sm font-semibold">
            {ruleTriggered}
          </span>
        </div>
      )}
    </div>
  )
}

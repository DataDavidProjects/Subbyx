import { FeatureDisplayProps } from "./FeatureDisplay.types"

interface FeatureMeta {
  label: string
  description: string
}

export function FeatureDisplay({ features }: FeatureDisplayProps) {
  if (!features) {
    return null
  }

  // Extract the metadata map and the actual feature entries
  const meta = (features["__meta"] ?? {}) as Record<string, FeatureMeta>
  const entries = Object.entries(features).filter(
    ([key, value]) => key !== "__meta" && value !== undefined && value !== null
  )

  if (entries.length === 0) {
    return null
  }

  return (
    <div className="rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-6 py-3 bg-gray-50 border-b border-gray-200">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-widest">
          Features ({entries.length})
        </p>
      </div>

      <div className="divide-y divide-gray-100">
        {entries.map(([key, value]) => {
          const info = meta[key] as FeatureMeta | undefined
          const label = info?.label || formatKey(key)
          const description = info?.description || ""

          return (
            <div
              key={key}
              className="px-6 py-3 flex items-start justify-between gap-6 hover:bg-gray-50 transition-colors group"
            >
              <div className="flex flex-col min-w-0 flex-1">
                {/* Human-readable label */}
                <span className="text-sm font-semibold text-gray-800">{label}</span>

                {/* Description — always shown, greyed out if missing */}
                <span
                  className={`text-xs mt-0.5 leading-snug ${
                    description ? "text-gray-500" : "text-gray-300 italic"
                  }`}
                >
                  {description || "No description available"}
                </span>

                {/* Raw feature key — subtle, for debugging */}
                <span className="text-[10px] font-mono text-gray-300 mt-1 group-hover:text-gray-400 transition-colors">
                  {key}
                </span>
              </div>

              {/* Value */}
              <span className="font-mono text-sm font-medium text-gray-900 shrink-0 mt-0.5 bg-gray-50 group-hover:bg-white px-2 py-0.5 rounded border border-gray-100 transition-colors">
                {formatValue(value)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function formatKey(key: string): string {
  return key
    .replace(/^.*__/, "") // strip view__ prefix
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function formatValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4)
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No"
  }
  if (value === null || value === undefined) {
    return "—"
  }
  return String(value)
}

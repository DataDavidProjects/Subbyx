import { FeatureDisplayProps } from "./FeatureDisplay.types"

export function FeatureDisplay({ features }: FeatureDisplayProps) {
  if (!features) {
    return null
  }

  const entries = Object.entries(features).filter(
    ([, value]) => value !== undefined && value !== null
  )

  if (entries.length === 0) {
    return null
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="px-6 py-4 border-b border-gray-100">
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
          Features ({entries.length})
        </p>
      </div>

      <div className="px-6 py-4 space-y-2">
        {entries.map(([key, value]) => (
          <div key={key} className="flex justify-between text-sm">
            <span className="text-gray-500">{formatKey(key)}</span>
            <span className="font-mono text-gray-900">{formatValue(value)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function formatKey(key: string): string {
  return key.replace(/_/g, " ")
}

function formatValue(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4)
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No"
  }
  return String(value)
}

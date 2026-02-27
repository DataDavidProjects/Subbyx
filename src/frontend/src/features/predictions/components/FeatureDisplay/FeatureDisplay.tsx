import { FeatureDisplayProps } from "./FeatureDisplay.types"

export function FeatureDisplay({ features }: FeatureDisplayProps) {
  if (!features) {
    return (
      <div className="text-gray-500 text-sm">No features available</div>
    )
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <h4 className="font-semibold text-gray-800 mb-4">Retrieved Features</h4>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FeatureGroup title="Customer" features={features.customer} />
        <FeatureGroup title="Charges" features={features.charges} />
        <FeatureGroup title="Payment Intents" features={features.payment_intents} />
        <FeatureGroup title="Checkout History" features={features.checkout_history} />
      </div>
    </div>
  )
}

function FeatureGroup({ title, features }: { title: string; features: Record<string, unknown> | undefined }) {
  if (!features) {
    return (
      <div className="p-3 bg-gray-50 rounded-lg">
        <h5 className="font-medium text-gray-700 text-sm">{title}</h5>
        <p className="text-gray-400 text-xs mt-1">No features</p>
      </div>
    )
  }

  const entries = Object.entries(features).filter(([, value]) => value !== undefined && value !== null)

  if (entries.length === 0) {
    return (
      <div className="p-3 bg-gray-50 rounded-lg">
        <h5 className="font-medium text-gray-700 text-sm">{title}</h5>
        <p className="text-gray-400 text-xs mt-1">No features</p>
      </div>
    )
  }

  return (
    <div className="p-3 bg-gray-50 rounded-lg">
      <h5 className="font-medium text-gray-700 text-sm mb-2">{title}</h5>
      <div className="space-y-1">
        {entries.map(([key, value]) => (
          <div key={key} className="flex justify-between text-xs">
            <span className="text-gray-500">{formatKey(key)}</span>
            <span className="font-mono text-gray-700">{formatValue(value)}</span>
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
    return value.toFixed(4)
  }
  return String(value)
}

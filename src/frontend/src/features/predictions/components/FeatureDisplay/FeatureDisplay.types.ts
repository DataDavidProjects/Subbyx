export interface FeatureDisplayProps {
  features: {
    customer?: Record<string, unknown>
    charges?: Record<string, unknown>
    payment_intents?: Record<string, unknown>
    checkout_history?: Record<string, unknown>
  } | null
}

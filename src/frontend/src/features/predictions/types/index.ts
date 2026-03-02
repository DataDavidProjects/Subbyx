export interface CheckoutData {
  subscription_value: number
  grade: string
  category: string
  has_linked_products: boolean
  has_vetrino: boolean
  has_protezione_totale: boolean
  has_protezione_furto: boolean
  store_id: string
  sku: string
  card_fingerprint?: string | null
}

export interface CheckoutResponse {
  decision: string
  reason: string
  rule_triggered: string | null
  score: number | null
  segment: string | null
  segment_reason: string | null
  features: Record<string, unknown> | null
  production_score: number | null
  shadow_score: number | null
  scored_by: string | null
}



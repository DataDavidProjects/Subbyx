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
}

export interface CheckoutResponse {
  decision: string
  reason: string
  rule_triggered: string | null
  score: number | null
  segment: string | null
  segment_reason: string | null
  features: {
    customer?: Record<string, unknown>
    charges?: Record<string, unknown>
    payment_intents?: Record<string, unknown>
    checkout_history?: Record<string, unknown>
  } | null
  production_score: number | null
  shadow_score: number | null
  canary_score: number | null
  scored_by: string | null
}

export interface BatchResponse {
  total: number
  blocked: number
  approved: number
  output_path: string
}

export interface PredictionResult {
  customer_id: string
  email: string
  segment: string | null
  score: number | null
  decision: string
  reason: string
  shadow_score: number | null
  canary_score: number | null
  scored_by: string | null
}

export interface CustomerFeatures {
  gender?: string
  card_owner_names_match_score?: number
  doc_name_email_match_score?: number
  email_emails_match_score?: number
  account_card_names_match_score?: number
  high_end_count?: number
  high_end_rate?: number
}

export interface ChargesFeatures {
  charge_count?: number
  charge_failure_rate?: number
  recurring_charge_rate?: number
}

export interface PaymentIntentFeatures {
  intent_count?: number
  total_failures?: number
  payment_failure_rate?: number
  avg_risk_score?: number
}

export interface CheckoutHistoryFeatures {
  prior_checkout_count?: number
  avg_subscription_value?: number
  distinct_categories?: number
}

export interface FeatureResponse {
  customer: CustomerFeatures
  charges: ChargesFeatures
  payment_intents: PaymentIntentFeatures
  checkout_history: CheckoutHistoryFeatures
}

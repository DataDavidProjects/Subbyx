export interface ResultCardProps {
  decision: string
  score: number | null
  segment: string | null
  reason: string
  ruleTriggered?: string | null
  segmentReason?: string | null
  productionScore?: number | null
  shadowScore?: number | null
  canaryScore?: number | null
  scoredBy?: string | null
}

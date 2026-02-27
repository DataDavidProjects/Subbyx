import { CheckoutData, CheckoutResponse, PredictionResult } from "../types"
import { logger } from "@/lib/logger"

const API_URL = typeof window !== "undefined" 
  ? (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001")
  : (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001")

export type Result<T> = {
  success: boolean
  data?: T
  error?: string
}

export interface BatchPredictionResult {
  total: number
  blocked: number
  approved: number
  results?: PredictionResult[]
}

export async function predictCheckoutClient(
  customerId: string,
  email: string,
  checkoutData: CheckoutData
): Promise<Result<CheckoutResponse>> {
  try {
    const response = await fetch(`${API_URL}/fraud/v1/checkout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        customer_id: customerId,
        email,
        checkout_data: checkoutData,
      }),
    })

    if (!response.ok) {
      const error = `Error: ${response.statusText}`
      return { success: false, error }
    }

    const data: CheckoutResponse = await response.json()
    return { success: true, data }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error"
    return { 
      success: false, 
      error: errorMessage 
    }
  }
}

export async function checkBlacklistClient(
  email: string
): Promise<Result<{ triggered: boolean; rule: string; reason: string }>> {
  try {
    const response = await fetch(`${API_URL}/fraud/v1/rules/blacklist/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    })

    if (!response.ok) {
      return { success: false, error: `Error: ${response.statusText}` }
    }

    const data = await response.json()
    return { success: true, data }
  } catch (error) {
    return { 
      success: false, 
      error: error instanceof Error ? error.message : "Unknown error" 
    }
  }
}

export async function predictBatchClient(file: File): Promise<Result<BatchPredictionResult>> {
  try {
    const formData = new FormData()
    formData.append("file", file)

    const response = await fetch(`${API_URL}/fraud/v1/batch/predict-file`, {
      method: "POST",
      body: formData,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      return { success: false, error: errorData.detail || `Error: ${response.statusText}` }
    }

    const data: BatchPredictionResult = await response.json()
    return { success: true, data }
  } catch (error) {
    return { 
      success: false, 
      error: error instanceof Error ? error.message : "Unknown error" 
    }
  }
}

"use server"

import { CheckoutData, CheckoutResponse } from "../types"
import { logger } from "@/lib/logger"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001"

export interface FutureCheckout {
  id: string
  customer_id: string
  email: string | null
  subscription_value: number
  grade: string
  category: string
  store_id: string
  sku: string
  has_linked_products: boolean
  has_vetrino: boolean
  has_protezione_totale: boolean
  has_protezione_furto: boolean
  mode: string
  status: string
  created: string
  card_fingerprint: string | null
}

export interface PaginatedCheckouts {
  checkouts: FutureCheckout[]
  total: number
  page: number
  pageSize: number
  totalPages: number
}

export async function loadFutureCheckouts(): Promise<FutureCheckout[]> {
  logger.info("[DAL] loadFutureCheckouts: fetching checkouts")
  
  const response = await fetch(
    `${API_URL}/fraud/v1/checkouts?mode=payment&status=complete`,
    { headers: { "Content-Type": "application/json" } }
  )

  if (!response.ok) {
    const err = `Failed to load checkouts: ${response.status}`
    logger.error({ status: response.status }, "[DAL] loadFutureCheckouts error")
    throw new Error(err)
  }

  const data = await response.json()
  const checkouts = data.checkouts || []
  
  logger.info({ count: checkouts.length }, "[DAL] loadFutureCheckouts: loaded checkouts")
  return checkouts
}

export interface CheckoutFilters {
  search?: string
  category?: string
  grade?: string
  sort_order?: "asc" | "desc"
}

export async function loadCheckoutsPaginated(
  page: number = 1,
  pageSize: number = 10,
  filters?: CheckoutFilters
): Promise<PaginatedCheckouts> {
  logger.info({ page, pageSize, filters }, "[DAL] loadCheckoutsPaginated: starting")
  
  const params = new URLSearchParams({
    mode: "payment",
    status: "complete",
    page: page.toString(),
    page_size: pageSize.toString(),
  })

  if (filters?.search) {
    params.append("search", filters.search)
  }
  if (filters?.category) {
    params.append("category", filters.category)
  }
  if (filters?.grade) {
    params.append("grade", filters.grade)
  }
  if (filters?.sort_order) {
    params.append("sort_order", filters.sort_order)
  }

  const response = await fetch(
    `${API_URL}/fraud/v1/checkouts?${params.toString()}`,
    { headers: { "Content-Type": "application/json" } }
  )

  if (!response.ok) {
    const err = `Failed to load checkouts: ${response.status}`
    logger.error({ status: response.status, params: params.toString() }, "[DAL] loadCheckoutsPaginated error")
    throw new Error(err)
  }

  const data = await response.json()
  const result = {
    checkouts: data.checkouts || [],
    total: data.total || 0,
    page: data.page || page,
    pageSize: data.page_size || pageSize,
    totalPages: data.total_pages || Math.ceil((data.total || 0) / pageSize),
  }
  
  logger.info({ 
    total: result.total, 
    page: result.page, 
    pageSize: result.pageSize,
    returned: result.checkouts.length 
  }, "[DAL] loadCheckoutsPaginated: loaded")
  
  return result
}

export async function predictCheckout(
  customerId: string,
  email: string,
  checkoutData: CheckoutData
): Promise<CheckoutResponse> {
  logger.info({ 
    customerId, 
    email, 
    checkoutValue: checkoutData.subscription_value,
    grade: checkoutData.grade,
    category: checkoutData.category
  }, "[DAL] predictCheckout: START - building request")

  const requestBody: Record<string, unknown> = {
    customer_id: customerId,
    email,
    checkout_data: checkoutData,
  }
  if (checkoutData.card_fingerprint) {
    requestBody.card_fingerprint = checkoutData.card_fingerprint
  }
  
  logger.debug({ requestBody }, "[DAL] predictCheckout: request body prepared")

  logger.info({ customerId, email }, "[DAL] predictCheckout: calling FastAPI backend")
  
  const response = await fetch(`${API_URL}/fraud/v1/checkout`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  })

  if (!response.ok) {
    const err = `Prediction failed: ${response.statusText}`
    logger.error({ 
      customerId, 
      email,
      status: response.status,
      statusText: response.statusText 
    }, "[DAL] predictCheckout: FastAPI error")
    throw new Error(err)
  }

  const data: CheckoutResponse = await response.json()
  
  logger.info({ 
    customerId,
    email,
    decision: data.decision,
    score: data.score,
    segment: data.segment,
    scoredBy: data.scored_by,
    productionScore: data.production_score,
    shadowScore: data.shadow_score
  }, "[DAL] predictCheckout: received response")
  
  logger.debug({ 
    features: data.features,
    segmentReason: data.segment_reason,
    reason: data.reason
  }, "[DAL] predictCheckout: full response details")

  return data
}

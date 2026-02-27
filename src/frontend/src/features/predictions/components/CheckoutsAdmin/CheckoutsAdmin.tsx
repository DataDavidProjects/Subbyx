"use client"

import { useState, useCallback } from "react"
import { toast } from "sonner"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CheckoutsTable } from "../CheckoutsTable/CheckoutsTable"
import { FutureCheckout, predictCheckout, loadCheckoutsPaginated, type CheckoutFilters } from "../../dal"
import { CheckoutResponse } from "../../types"
import { ResultCard } from "../ResultCard/ResultCard"
import { FeatureDisplay } from "../FeatureDisplay/FeatureDisplay"
import { logger } from "@/lib/logger"

const LOG_PREFIX = "[CheckoutsAdmin]"

export interface CheckoutsAdminProps {
  initialData: {
    checkouts: FutureCheckout[]
    total: number
    page: number
    pageSize: number
    totalPages: number
  }
}

export function CheckoutsAdmin({ initialData }: CheckoutsAdminProps) {
  const [checkouts, setCheckouts] = useState(initialData.checkouts)
  const [total, setTotal] = useState(initialData.total)
  const [page, setPage] = useState(initialData.page)
  const [pageSize, setPageSize] = useState(initialData.pageSize)
  const [totalPages, setTotalPages] = useState(initialData.totalPages)
  const [isLoading, setIsLoading] = useState(false)
  const [selectedCheckout, setSelectedCheckout] = useState<FutureCheckout | null>(null)
  const [prediction, setPrediction] = useState<CheckoutResponse | null>(null)
  const [predictionError, setPredictionError] = useState<string | null>(null)
  const [isPredicting, setIsPredicting] = useState(false)
  const [filters, setFilters] = useState<CheckoutFilters>({})
  const [activeTab, setActiveTab] = useState("browse")

  const fetchCheckouts = useCallback(async (newPage: number, newFilters?: CheckoutFilters) => {
    setIsLoading(true)
    try {
      const activeFilters = newFilters ?? filters
      const data = await loadCheckoutsPaginated(newPage, pageSize, activeFilters)
      setCheckouts(data.checkouts)
      setTotal(data.total)
      setPage(data.page)
      setTotalPages(data.totalPages)
    } catch (error) {
      logger.error({ error }, "Failed to fetch checkouts")
    } finally {
      setIsLoading(false)
    }
  }, [pageSize, filters])

  const handlePageChange = (newPage: number) => {
    fetchCheckouts(newPage)
  }

  const handleSearch = (search: string) => {
    const newFilters = { ...filters, search: search || undefined }
    setFilters(newFilters)
    fetchCheckouts(1, newFilters)
  }

  const handleFilterChange = (newFilters: CheckoutFilters) => {
    const merged = { ...filters, ...newFilters }
    setFilters(merged)
    fetchCheckouts(1, merged)
  }

  const handleSelect = (checkout: FutureCheckout | null) => {
    setSelectedCheckout(checkout)
    setPrediction(null)
    setPredictionError(null)
  }

  const handlePredict = async (checkout: FutureCheckout) => {
    setIsPredicting(true)
    setPrediction(null)
    setPredictionError(null)

    try {
      const checkoutData = {
        subscription_value: checkout.subscription_value,
        grade: checkout.grade,
        category: checkout.category,
        store_id: checkout.store_id,
        sku: checkout.sku,
        has_linked_products: checkout.has_linked_products,
        has_vetrino: checkout.has_vetrino,
        has_protezione_totale: checkout.has_protezione_totale,
        has_protezione_furto: checkout.has_protezione_furto,
      }

      const result = await predictCheckout(checkout.customer_id, checkout.email || "", checkoutData)

      setPrediction(result)
      setActiveTab("prediction")

      const toastDescription = result.rule_triggered
        ? `Decision: ${result.decision} — Rule: ${result.rule_triggered}`
        : `Decision: ${result.decision} — Score: ${(result.score ?? 0).toFixed(2)}`
      const isApproved = result.decision === "APPROVE"
      const toastFn = isApproved ? toast.success : toast.error
      toastFn("Prediction complete", { description: toastDescription })
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error"
      setPredictionError(message)
      toast.error("Prediction failed", { description: message })
      logger.error({ checkoutId: checkout.id, error: message }, "Prediction failed")
    } finally {
      setIsPredicting(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Checkouts Management</h2>
          <p className="text-muted-foreground">
            Select a checkout to run fraud prediction analysis
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>{total} total checkouts</span>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full max-w-sm grid-cols-2">
          <TabsTrigger value="browse">Browse Checkouts</TabsTrigger>
          <TabsTrigger value="prediction">
            {prediction ? "Result" : "Prediction"}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="browse" className="space-y-4">
          <CheckoutsTable
            checkouts={checkouts}
            total={total}
            page={page}
            pageSize={pageSize}
            totalPages={totalPages}
            isLoading={isLoading}
            isPredicting={isPredicting}
            selectedId={selectedCheckout?.id || null}
            onSelect={handleSelect}
            onPredict={handlePredict}
            onPageChange={handlePageChange}
            onSearch={handleSearch}
            onFilterChange={handleFilterChange}
          />
        </TabsContent>

        <TabsContent value="prediction">
          <Card>
            <CardHeader>
              <CardTitle>Fraud Prediction</CardTitle>
            </CardHeader>
            <CardContent>
              {isPredicting ? (
                <div className="flex items-center justify-center py-8">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4" />
                    <p className="text-muted-foreground">Running prediction...</p>
                  </div>
                </div>
              ) : predictionError ? (
                <div className="bg-destructive/10 border border-destructive rounded-lg p-4">
                  <p className="text-destructive font-medium">Prediction Failed</p>
                  <p className="text-destructive/80 text-sm mt-1">{predictionError}</p>
                </div>
              ) : prediction ? (
                <div className="space-y-6">
                  <ResultCard
                    decision={prediction.decision}
                    score={prediction.score}
                    segment={prediction.segment}
                    reason={prediction.reason}
                    ruleTriggered={prediction.rule_triggered}
                    segmentReason={prediction.segment_reason}
                    shadowScore={prediction.shadow_score}
                    canaryScore={prediction.canary_score}
                    scoredBy={prediction.scored_by}
                  />
                  <FeatureDisplay features={prediction.features} />
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <p>No prediction run yet.</p>
                  <p className="text-sm">Select a checkout and click &quot;Run Prediction&quot; to analyze.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

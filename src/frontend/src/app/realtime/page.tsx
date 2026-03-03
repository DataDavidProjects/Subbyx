import { Suspense } from "react"

import { loadCheckoutsPaginated } from "@/features/predictions/dal"
import { CheckoutsAdmin } from "@/features/predictions/components"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"

export const dynamic = "force-dynamic"

async function AdminContent() {
  const initialData = await loadCheckoutsPaginated(1, 10)
  return <CheckoutsAdmin initialData={initialData} />
}

function AdminSkeleton() {
  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-4 w-96" />
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Browse Checkouts</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Skeleton className="h-10 w-64" />
          <Skeleton className="h-64 w-full" />
          <div className="flex justify-between">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-8 w-48" />
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default function RealtimePage() {
  return (
    <div className="container mx-auto py-8 px-4">
      <Suspense fallback={<AdminSkeleton />}>
        <AdminContent />
      </Suspense>
    </div>
  )
}

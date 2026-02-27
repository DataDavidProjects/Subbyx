import { FutureCheckout } from "../../dal"

export interface CheckoutsAdminProps {
  initialData: {
    checkouts: FutureCheckout[]
    total: number
    page: number
    pageSize: number
    totalPages: number
  }
}

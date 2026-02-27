import { FutureCheckout, type CheckoutFilters } from "../../dal"

export interface CheckoutsTableProps {
  checkouts: FutureCheckout[]
  total: number
  page: number
  pageSize: number
  totalPages: number
  isLoading?: boolean
  isPredicting?: boolean
  selectedId?: string | null
  onSelect: (checkout: FutureCheckout | null) => void
  onPredict: (checkout: FutureCheckout) => void
  onPageChange: (page: number) => void
  onSearch: (search: string) => void
  onFilterChange: (filters: CheckoutFilters) => void
}

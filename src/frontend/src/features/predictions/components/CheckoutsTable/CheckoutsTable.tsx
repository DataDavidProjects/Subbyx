"use client"

import { useState, useRef, useEffect, useCallback } from "react"

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Checkbox } from "@/components/ui/checkbox"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Copy, Check, ArrowUpDown } from "lucide-react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination"
import { Skeleton } from "@/components/ui/skeleton"
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

const CATEGORY_OPTIONS = [
  "laptops",
  "tablets",
  "tv",
  "smartphones",
  "gps_navigator",
  "audio",
  "gaming",
  "wearable",
]

const GRADE_OPTIONS = ["A", "B", "C", "new"]

export function CheckoutsTable({
  checkouts,
  total,
  page,
  pageSize,
  totalPages,
  isLoading,
  isPredicting,
  selectedId,
  onSelect,
  onPredict,
  onPageChange,
  onSearch,
  onFilterChange,
}: CheckoutsTableProps) {
  const [searchValue, setSearchValue] = useState("")
  const [categoryFilter, setCategoryFilter] = useState<string>("")
  const [gradeFilter, setGradeFilter] = useState<string>("")
  const [truthFilter, setTruthFilter] = useState<string>("all")
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc")

  const handleSearch = () => {
    onSearch(searchValue)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch()
    }
  }

  const handleCategoryChange = (value: string) => {
    const category = value === "all" ? "" : value
    setCategoryFilter(category)
    onFilterChange({ category: category || undefined })
  }

  const handleGradeChange = (value: string) => {
    const grade = value === "all" ? "" : value
    setGradeFilter(grade)
    onFilterChange({ grade: grade || undefined })
  }

  const handleTruthChange = (value: string) => {
    setTruthFilter(value)
    const is_fraud = value === "fraud" ? true : value === "clean" ? false : undefined
    onFilterChange({ is_fraud })
  }

  const handleSortToggle = () => {
    const next = sortOrder === "desc" ? "asc" : "desc"
    setSortOrder(next)
    onFilterChange({ sort_order: next })
  }

  const handleCopyRow = async (checkout: FutureCheckout) => {
    const rowData = {
      id: checkout.id,
      customer_id: checkout.customer_id,
      email: checkout.email,
      subscription_value: checkout.subscription_value,
      category: checkout.category,
      grade: checkout.grade,
      sku: checkout.sku,
      created: checkout.created,
    }
    await navigator.clipboard.writeText(JSON.stringify(rowData, null, 2))
    setCopiedId(checkout.id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString("en-GB", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    })
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-EU", {
      style: "currency",
      currency: "EUR",
    }).format(value)
  }

  const topScrollRef = useRef<HTMLDivElement>(null)
  const tableScrollRef = useRef<HTMLDivElement>(null)
  const [scrollWidth, setScrollWidth] = useState(0)

  const syncScroll = useCallback((source: "top" | "table") => {
    const top = topScrollRef.current
    const table = tableScrollRef.current
    if (!top || !table) return
    if (source === "top") {
      table.scrollLeft = top.scrollLeft
    } else {
      top.scrollLeft = table.scrollLeft
    }
  }, [])

  useEffect(() => {
    const table = tableScrollRef.current
    if (table) {
      const updateWidth = () => setScrollWidth(table.scrollWidth)
      updateWidth()
      const observer = new ResizeObserver(updateWidth)
      observer.observe(table)
      return () => observer.disconnect()
    }
  }, [checkouts])

  const selectedCheckout = checkouts.find((c) => c.id === selectedId)
  const startItem = (page - 1) * pageSize + 1
  const endItem = Math.min(page * pageSize, total)

  return (
    <div className="space-y-4">
      {/* Selected checkout bar - shown at top */}
      {selectedCheckout && (
        <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg border">
          <div className="flex flex-col">
            <span className="text-sm font-medium">Selected Checkout</span>
            <span className="text-sm text-muted-foreground">
              {selectedCheckout.email} &bull; {formatCurrency(selectedCheckout.subscription_value)}
            </span>
          </div>
          <Button
            onClick={() => onPredict(selectedCheckout)}
            disabled={isPredicting}
          >
            {isPredicting && (
              <span className="mr-2 inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
            )}
            {isPredicting ? "Running..." : "Run Prediction"}
          </Button>
        </div>
      )}

      {/* Search and filters */}
      <div className="flex flex-wrap items-center gap-2">
        <Input
          placeholder="Search by email, customer ID or SKU..."
          value={searchValue}
          onChange={(e) => setSearchValue(e.target.value)}
          onKeyDown={handleKeyDown}
          className="max-w-sm"
        />
        <Button variant="outline" onClick={handleSearch}>
          Search
        </Button>
        <Select value={categoryFilter || "all"} onValueChange={handleCategoryChange}>
          <SelectTrigger className="w-[160px]">
            <SelectValue placeholder="Category" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All categories</SelectItem>
            {CATEGORY_OPTIONS.map((cat) => (
              <SelectItem key={cat} value={cat}>
                {cat}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={gradeFilter || "all"} onValueChange={handleGradeChange}>
          <SelectTrigger className="w-[120px]">
            <SelectValue placeholder="Grade" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All grades</SelectItem>
            {GRADE_OPTIONS.map((g) => (
              <SelectItem key={g} value={g}>
                {g}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select value={truthFilter} onValueChange={handleTruthChange}>
          <SelectTrigger className="w-[120px]">
            <SelectValue placeholder="Truth" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Truth</SelectItem>
            <SelectItem value="fraud">Fraud</SelectItem>
            <SelectItem value="clean">Clean</SelectItem>
          </SelectContent>
        </Select>
        <Button variant="outline" onClick={handleSortToggle} className="gap-1.5">
          <ArrowUpDown className="h-4 w-4" />
          Date {sortOrder === "desc" ? "Newest" : "Oldest"}
        </Button>
      </div>

      {/* Top scrollbar */}
      <div
        ref={topScrollRef}
        className="overflow-x-auto"
        onScroll={() => syncScroll("top")}
      >
        <div style={{ width: scrollWidth, height: 1 }} />
      </div>

      {/* Scrollable table */}
      <div
        ref={tableScrollRef}
        className="rounded-md border overflow-x-auto"
        onScroll={() => syncScroll("table")}
      >
        <Table className="min-w-[900px]">
          <TableHeader>
            <TableRow>
              <TableHead className="w-12">Select</TableHead>
              <TableHead className="w-12">Copy</TableHead>
              <TableHead>ID</TableHead>
              <TableHead>Customer</TableHead>
              <TableHead>Email</TableHead>
              <TableHead>Value</TableHead>
              <TableHead>Category</TableHead>
              <TableHead>Grade</TableHead>
              <TableHead>SKU</TableHead>
              <TableHead>Truth</TableHead>
              <TableHead>Created</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <TableRow key={i}>
                  <TableCell><Skeleton className="h-4 w-4" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-8" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-20" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-24" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-40" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-20" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-20" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-12" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-24" /></TableCell>
                  <TableCell><Skeleton className="h-4 w-32" /></TableCell>
                </TableRow>
              ))
            ) : checkouts.length === 0 ? (
              <TableRow>
                <TableCell colSpan={10} className="h-24 text-center">
                  No checkouts found.
                </TableCell>
              </TableRow>
            ) : (
              checkouts.map((checkout) => (
                <TableRow
                  key={checkout.id}
                  className={`cursor-pointer ${
                    selectedId === checkout.id ? "bg-muted" : ""
                  }`}
                  onClick={() => onSelect(checkout)}
                >
                  <TableCell onClick={(e) => e.stopPropagation()}>
                    <Checkbox
                      checked={selectedId === checkout.id}
                      onCheckedChange={() => onSelect(checkout)}
                    />
                  </TableCell>
                  <TableCell onClick={(e) => e.stopPropagation()}>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleCopyRow(checkout)}
                      title="Copy row data"
                    >
                      {copiedId === checkout.id ? (
                        <Check className="h-4 w-4 text-green-500" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </Button>
                  </TableCell>
                  <TableCell className="font-mono text-xs whitespace-nowrap">
                    {checkout.id.slice(0, 8)}...
                  </TableCell>
                  <TableCell className="font-medium whitespace-nowrap">
                    {checkout.customer_id}
                  </TableCell>
                  <TableCell className="whitespace-nowrap">{checkout.email}</TableCell>
                  <TableCell className="whitespace-nowrap">{formatCurrency(checkout.subscription_value)}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{checkout.category}</Badge>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={
                        checkout.grade === "A"
                          ? "default"
                          : checkout.grade === "B"
                          ? "secondary"
                          : "destructive"
                      }
                    >
                      {checkout.grade}
                    </Badge>
                  </TableCell>
                  <TableCell className="font-mono text-xs whitespace-nowrap">
                    {checkout.sku}
                  </TableCell>
                  <TableCell>
                    <Badge variant={checkout.is_fraud ? "destructive" : "outline"}>
                      {checkout.is_fraud ? "FRAUD" : "CLEAN"}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground whitespace-nowrap">
                    {formatDate(checkout.created)}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          Showing {startItem}–{endItem} of {total} checkouts
        </p>
        <Pagination>
          <PaginationContent>
            <PaginationItem>
              <PaginationPrevious
                href="#"
                onClick={(e) => {
                  e.preventDefault()
                  if (page > 1) onPageChange(page - 1)
                }}
                className={page <= 1 ? "pointer-events-none opacity-50" : ""}
              />
            </PaginationItem>
            <PaginationItem>
              <span className="px-4 py-2 text-sm">
                Page {page} of {totalPages}
              </span>
            </PaginationItem>
            <PaginationItem>
              <PaginationNext
                href="#"
                onClick={(e) => {
                  e.preventDefault()
                  if (page < totalPages) onPageChange(page + 1)
                }}
                className={page >= totalPages ? "pointer-events-none opacity-50" : ""}
              />
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      </div>
    </div>
  )
}

# Subbyx - Fraud Detection System

A real-time fraud detection system for checkout transactions using FastAPI (backend) and Next.js (frontend).

## Project Structure

```
src/
├── backend/          # FastAPI fraud detection API (Python 3.14)
├── frontend/         # Next.js prediction UI (Node.js 20+)
├── scripts/         # Data processing scripts
data/
├── 00-raw/          # Raw CSV data
├── 01-clean/        # Cleaned data
├── 02-elaboration/  # Train/test splits
└── 03-processed/    # Parquet files
```

## Quick Start

### Docker (all services)
```bash
make up

# Open browser
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000/docs
# MLflow:   http://localhost:5002
```

### Local Development (no Docker)
```bash
# Terminal 1: Start backend + MLflow
make dev-backend

# Terminal 2: Start frontend
make dev-frontend
```

## Services

| Service | Port (Docker) | Port (Local) | Description |
|---------|---------------|--------------|-------------|
| Frontend | 3000 | 3001 | Next.js prediction UI |
| Backend | 8000 | 8001 | FastAPI fraud detection API |
| Redis | 6379 | - | Feature store online store |
| MLflow | 5002 | 5002 | Model tracking |

## API Endpoints

- `POST /fraud/v1/checkout` - Predict fraud for a checkout
- `POST /fraud/v1/segment/determine` - Determine customer segment
- `POST /fraud/v1/features/get` - Get customer features
- `POST /fraud/v1/decision` - Get decision based on score
- `POST /fraud/v1/score` - Get model score for features
- `POST /fraud/v1/batch/predict` - Batch prediction from CSV

All scoring endpoints (`/checkout`, `/score`, `/batch/predict`) return shadow and canary model scores alongside the production score. See [Shadow & Canary Scoring](#shadow--canary-scoring) below.

## Commands

```bash
# Docker
make up              # Start all services
make down            # Stop all services
make restart         # Restart services
make logs            # View logs

# Local development
make dev-backend     # Backend on port 8001 + MLflow on 5002
make dev-frontend    # Frontend on port 3001

# Other
make lint            # Run linters
make format          # Format code
```

## Shadow & Canary Scoring

The system supports scoring each request with multiple models simultaneously for safe model rollouts:

- **Shadow model** (always-on): every request is also scored by a shadow model (`@shadow` MLflow alias). The shadow score is logged and returned in the response but never used for decisions.
- **Canary model** (traffic-gated): a configurable percentage of traffic is scored by a canary model (`@canary` MLflow alias). When active, the canary score is used for the actual decision; the production score is still computed for comparison.

Configuration lives in `src/backend/services/fraud/inference/config.yaml`:

```yaml
shadow:
  enabled: true
  model_uri: models:/fraud-detector@shadow

canary:
  enabled: false
  model_uri: models:/fraud-detector@canary
  traffic_percentage: 10   # % of requests scored by canary
```

API responses include additional fields: `production_score`, `shadow_score`, `canary_score`, and `scored_by` (either `"production"` or `"canary"`).

---

# Dataset Documentation

## Table: customers

The customer object is created each time an unregistered user begins the subscription process for a plan. It is later associated with an email. Therefore, it is possible for the same email to be associated with two or more customers. In this case, the user is only one but with multiple associated customers. All subscriptions for these customers are grouped under the single email.

| Column | Explanation | Notes |
|--------|--------------|-------|
| id | Customer ID | A single individual may be associated with multiple customers |
| email | Anonymized customer email | |
| created | Customer creation date | |
| fiscal_code | Anonymized fiscal code | |
| birth_date | Customer birth date | Calculated from the fiscal code |
| residential_address_id | Residential address ID | |
| shipping_address_id | Shipping address ID | |
| card_owner_names_card_owner_names_match_score | Similarity score between names on credit cards used by the individual associated with the customer and any other credit card used by a different individual | |
| doc_name_email_match_score | Similarity score between the name extracted from the identity document and the email | |
| email_emails_match_score | Similarity score between the customer email and any other email entered | |
| account_card_names_match_score | Similarity score between the account name and the name on the card | |
| high_end_count | Number of high-end devices purchased | |
| high_end_rate | Ratio of high-end devices purchased to total devices purchased | |
| dunning_days | Number of days since a payment was missed by the customer | |

---

## Table: checkouts

The checkout table represents **subscription requests** - the act of a customer requesting to subscribe to a device. It is NOT the monthly payment installments.

**Checkout modes:**
- `mode=setup` - Customer registration (dummy record, no device). Created when a customer account is created.
- `mode=payment` - **Real subscription request** (the actual subscription to a device). Contains device details like subscription_value, grade, category, sku, etc.
- `mode=subscription` - **Real subscription request**, equivalent to `mode=payment`.

**Important:** Only `mode=payment` and `mode=subscription` rows contain real subscription data. The `mode=setup` rows have no device information (subscription_value is empty).

**Customer journey:**
1. Customer starts registration → `mode=setup` (account created)
2. Customer subscribes to device → `mode=payment` or `mode=subscription` (real subscription - this is what we predict!)
3. Monthly installments → Tracked in `charges` and `payment_intents` tables

**Prediction:** Given a new subscription request (`mode=payment` or `mode=subscription`), will the customer end up in dunning (>15 days late on payments)?

| Column | Explanation | Notes |
|--------|--------------|-------|
| id | Checkout ID made by the customer | |
| created | Creation date | |
| status | Checkout status | `complete` = successful, `expired` = failed/abandoned |
| payment_intent | Associated payment intent | Only populated for `mode=payment`/`subscription` |
| mode | Checkout mode | `setup`=registration, `payment`=subscription request |
| customer | Associated customer ID | |
| subscription_value | Monthly installment amount | Only populated for `mode=payment`/`subscription` |
| grade | Device wear grade | Only for `mode=payment`/`subscription` |
| store_id | Store ID where the subscription was made | |
| sku | Model identifier | Only for `mode=payment`/`subscription` |
| has_linked_products | Presence of products associated with the main product | Associated products can be tablet pens, theft covers, or other types of covers |
| has_vetrino | Presence of screen protector | |
| has_protezione_totale | Presence of total protection warranty | |
| has_protezione_furto | Presence of theft coverage | |
| product_description | Product description | Only for `mode=payment`/`subscription` |
| email | Associated customer email | |
| category | Product category | Only for `mode=payment`/`subscription` |
| condition | Product condition | Only for `mode=payment`/`subscription` | |

---

## Table: payment_intents

A payment intent is created each time a payment is due from the customer. If an installment is not paid, multiple successive payment intents can be created for that same installment. Usually, a payment intent is created each day until the installment is paid.

| Column | Explanation | Notes |
|--------|--------------|-------|
| id | Payment intent ID | |
| amount | Payment amount due | |
| amount_received | Payment amount received | |
| canceled_at | Payment attempt cancellation date | |
| cancellation_reason | Cancellation reason | |
| created | Payment intent creation date | |
| customer | Associated customer ID | |
| latest_charge | Latest associated charge | |
| status | Payment status | |
| payment_error_code | Payment error code | |
| subscription_value | Installment value | The installment is calculated from the product value and is the same for all months of the year. An exception is only the second month of subscription. Indeed, the second installment equals the sum due to pay the product from the day payment is created to the 1st of the following month. For example, if a user subscribes on January 10 and the installment is 50 euros, they will pay 50 euros on January 12, 32.14 euros on February 12, and from then on 50 euros on the 1st of each subsequent month. |
| n_failures | Number of payment failures | |

---

## Table: charges

A charge is created each time a debit attempt is made. This attempt is made starting from a payment intent generated earlier. Each charge is associated with a customer and a payment intent.

| Column | Explanation | Notes |
|--------|--------------|-------|
| id | Charge ID | |
| created | Creation date | |
| customer | Associated customer ID | |
| failure_code | Failure code | |
| paid | Successful payment | |
| payment_intent | Associated payment intent ID | |
| status | Charge status | |
| email | Associated customer email | |
| is_recurrent | Indicates if the payment is recurring | If it is the first payment, i.e., the one made at the time of subscription, this value will be FALSE, otherwise TRUE |
| outcome_status | Charge outcome | |
| outcome_risk | Payment risk levels | These values come directly from the Stripe service we use to process payments |
| outcome_risk_level | | |
| outcome_risk_score | | |
| outcome_type | Outcome type based on result | |
| card_fingerprint | Used card ID | |
| card_brand | Card circuit | |
| card_funding | Card type | |
| card_issuer | Card issuer | |
| card_cvc_check | CVC code check | |

---

## Table: addresses

Addresses are created from the addresses entered by users during purchase.

| Column | Explanation | Notes |
|--------|--------------|-------|
| id | Anonymized address ID | |
| locality | Locality | |
| city | City | |
| state | Region | |
| administrative_area_level_1 | Main administrative area | |
| country | Country | |
| postal_code | Postal code | |

---

## Table: sellers

Sellers are the physical and non-physical stores through which subscriptions are sold.

| Column | Explanation | Notes |
|--------|--------------|-------|
| name | Seller name | |
| store_id | Store ID | |
| partner_name | Partner name | |
| store_name | Store name | |
| address | Complete store address | |
| zip | Postal code | |
| state | Region | |
| province | Province | |
| area | Geographic area | |

---

## Business Notes

The goal is to create an anti-fraud system using the approach of your choice.

I recommend starting from the `customers` table, which contains the main user information. In particular, the `dunning_days` column indicates the delay in payments, the data we want to predict. Usually, we consider a user as a potential fraudster if the value exceeds 15 days, but you are free to choose the threshold you prefer.

The other tables can be linked to the `customers` table through the ID columns present in each table.


Documentation reference:
/Users/davidelupis/Desktop/Subbyx/docs

Feast:
https://github.com/feast-dev/feast/tree/master/docs/getting-started/concepts


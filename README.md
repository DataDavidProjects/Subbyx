# Subbyx - Fraud Detection System

A real-time fraud detection system for checkout transactions.


## Quick Start

### Docker (all services)
```bash
make up
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
| Frontend | 3001 | 3001 | Next.js prediction UI |
| Backend | 8001 | 8001 | FastAPI fraud detection API |
| Redis | 6379 | - | Feature store online store |
| MLflow | 5002 | 5002 | Model tracking |


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

## Documentation reference:
/Users/davidelupis/Desktop/Subbyx/docs

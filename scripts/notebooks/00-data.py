import pandas as pd


DATA_PATH = "/Users/davidelupis/Desktop/Subbyx/data/01-clean/"

customers = pd.read_csv(DATA_PATH + "customers.csv")
addresses = pd.read_csv(DATA_PATH + "addresses.csv")
charges = pd.read_csv(DATA_PATH + "charges.csv")
checkouts = pd.read_csv(DATA_PATH + "checkouts.csv")
payment_intents = pd.read_csv(DATA_PATH + "payment_intents.csv")
stores = pd.read_csv(DATA_PATH + "stores.csv")


# Customer with a fraud
print(customers.head())

print(addresses.head())
print(charges.head())
print(checkouts.head())
print(payment_intents.head())
print(stores.head())


print(customers.shape)
print(addresses.shape)
print(charges.shape)
print(checkouts.shape)
print(payment_intents.shape)
print(stores.shape)


print(customers.isnull().sum())
print(addresses.isnull().sum())
print(charges.isnull().sum())
print(checkouts.isnull().sum())
print(payment_intents.isnull().sum())
print(stores.isnull().sum())


codice_fiscale_email = (
    customers.groupby("fiscal_code").agg({"id": "nunique"}).sort_values(by="id", ascending=False)
)
print(codice_fiscale_email.loc[codice_fiscale_email["id"] > 1])

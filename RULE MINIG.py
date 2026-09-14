import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

transactions = [
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter"],
    ["Milk", "Bread"],
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter", "Jam"],
    ["Milk", "Bread", "Jam"],
    ["Milk", "Butter"],
    ["Bread", "Butter"],
    ["Milk", "Bread", "Butter"],
    ["Bread", "Jam"]
]

encoder = TransactionEncoder()

encoded_data = encoder.fit(transactions).transform(transactions)

df = pd.DataFrame(
    encoded_data,
    columns=encoder.columns_
)

print("Transaction Dataset:")
print(df)

frequent_itemsets = apriori(
    df,
    min_support=0.3,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

rules = rules[
    ["antecedents", "consequents", "support", "confidence", "lift"]
]

print("\nAssociation Rules:")
print(rules)

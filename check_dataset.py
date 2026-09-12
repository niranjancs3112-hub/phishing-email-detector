import pandas as pd

data = pd.read_csv("phishing_email.csv")

print("Label counts:")
print(data["label"].value_counts())

print("\nSample emails for each label:")

print("\nLabel 0:")
print(data[data["label"] == 0]["text_combined"].head(3))

print("\nLabel 1:")
print(data[data["label"] == 1]["text_combined"].head(3))
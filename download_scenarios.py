import deepmimo as dm

# Define search parameters
query = {
    'bands': ['sub6',],
    'environment': 'outdoor',
    # 'numRx': {'min': 10e3, 'max': 10e5}
}

# Perform search
scenarios = dm.search(query)  # returns ['scenario1', 'scenario2', ...]
print(scenarios)
# # Can be used later to download scenarios systematically
## failed at city_11_villa_florida_1gp_1758180246123
for scenario in scenarios:
    dm.download(scenario)
    print(f"Downloaded scenario: {scenario}")
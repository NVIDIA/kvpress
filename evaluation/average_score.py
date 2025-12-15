import json

# Load the metrics
with open('results/ruler__4096__meta-llama--Meta-Llama-3.1-8B-Instruct__kvsquared__0.90__fraction0.010/metrics.json', 'r') as f:
    metrics = json.load(f)

# Calculate average across all tasks
scores = [task_data['string_match'] for task_data in metrics.values()]
average_score = sum(scores) / len(scores)

print(f"Average RULER score: {average_score:.2f}")
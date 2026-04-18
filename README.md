CLI USAGE
```
# Run the offline demo (no API keys needed)
uv run python main.py demo

# Route a task and inspect the decision
uv run python main.py route --task "Implement quicksort" --difficulty hard --ground-truth

# Compare both strategies on a task
uv run python main.py compare --task "Explain merge sort" --trajectories 3

# Run the test suite (148 tests)
uv run python main.py test
```

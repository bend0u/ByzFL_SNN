import re

with open("byzfl/benchmark/data.py", "r") as f:
    content = f.read()

# Remove the shared memory cache logic from load_and_split_data
content = re.sub(
    r'\s*# Check if we have pre-loaded shared memory tensors available.*?(?=train_loader = torch\.utils\.data\.DataLoader)',
    '\n        ',
    content,
    flags=re.DOTALL
)

with open("byzfl/benchmark/data.py", "w") as f:
    f.write(content)

with open("byzfl/benchmark/benchmark.py", "r") as f:
    content = f.read()

# Remove pre_load_shared_dataset from benchmark.py
content = re.sub(
    r'\s*shared_cache = pre_load_shared_dataset.*?params_manager\.params\["shared_dataset_cache"\] = shared_cache',
    '',
    content,
    flags=re.DOTALL
)

with open("byzfl/benchmark/benchmark.py", "w") as f:
    f.write(content)


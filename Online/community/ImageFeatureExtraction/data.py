import kagglehub

# Download latest version
path = kagglehub.dataset_download("ipythonx/ade20k-scene-parsing")

print("Path to dataset files:", path)
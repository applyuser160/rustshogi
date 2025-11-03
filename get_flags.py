import pyo3

print(" ".join(pyo3.get_config().get_rust_flags()))

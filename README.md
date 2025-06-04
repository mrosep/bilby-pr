# bilby-pr

Posterior repartitioning in bilby

## Installation

Clone the repo and install the code using pip:

```
pip install .
```

You may want to use `-e` to allow for an editable install


## Usage

When calling `run_sampler`:

```python
bilby.run_sampler(
    sampler="dynesty-pr",
    weights_file=<path/to/weights/file.pkl>
    ...
)

```

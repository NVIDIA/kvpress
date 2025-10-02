# Needle in a Haystack
This benchmark evaluates a model's ability to retrieve a specific piece of information, the "needle," hidden within a large body of text, the "haystack." The test challenges a model's long-context understanding and its ability to maintain information accuracy over increasing document lengths. 
We follow the vast majority of the literature and use [Paul Graham's essays](https://huggingface.co/datasets/alessiodevoto/paul_graham_essays) as the haystack.

> The default needle is a sentence defined in the dataset itself, but it can be replaced by a custom sentence (e.g. for the passkey retrieval or similar tests). To do that, check [utils.py](./utils.py).
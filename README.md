# Ordered Khasi embeddings and surrogate controls

A within-corpus diagnostic of word order using fixed 32-dimensional CBOW vectors. Forty documents are selected deterministically by content hash from 770 eligible texts. Each contributes the longest contiguous known-token segment, truncated to 512 tokens; the minimum is 128. Unknown tokens break segments rather than being silently bridged.

The analysis compares adjacent-token cosine similarity with 99 word-order permutations. A second diagnostic projects normalized word vectors onto their first principal component and compares time asymmetry with 99 IAAFT surrogates. Each surrogate preserves the marginal distribution exactly. Spectral approximation must meet the predefined 5% relative-error threshold before a document's test is interpreted. Benjamini–Hochberg correction covers all 40 planned tests; failed-quality cases receive p=1 for multiplicity correction.

## Results

Mean adjacent-cosine difference versus permutation: 0.05479; descriptive document-bootstrap interval [0.04498, 0.06433]. Local co-occurrence is already part of CBOW training, so this is not independent evidence of language understanding. Documents may share publishers and need not be independent.

Only 8 of 40 documents pass surrogate quality checks. No time-asymmetry rejection survives correction across the planned family. The nonlinear-dynamics question remains inconclusive under this protocol. Neither deterministic chaos nor a cross-language difference is established.

This is an independent NumPy implementation informed by the TISEAN methodology, not execution of the TISEAN binary or restoration of a historical English/Russian experiment. The first diagnostic output is preserved separately; the reported results use correction over all planned documents.

## Reproduce

```text
python analyse.py --archive path/to/khasi_required_final.zip
python -m pytest -q test_surrogates.py
```

The archive must contain the original a `raw_texts_by_title/` folder of texts and `khasi_required_final/4_embeddings/cbow/khasi_cbow_word_vectors_dim32.csv`. Actual archive paths are checked by the script. Raw texts are not included for publication. See `results/protocol.json`, `document_results.csv` and `summary.json` for parameters and measurements.

Method reference: https://www.pks.mpg.de/tisean/TISEAN_2.1/docs/surropaper/node12.html

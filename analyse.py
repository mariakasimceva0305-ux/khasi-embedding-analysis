"""Ordered Khasi embeddings versus permutation and IAAFT controls.
Independent NumPy implementation; not a rerun of an unavailable historical TISEAN job.
"""

from pathlib import Path
from zipfile import ZipFile
import io, json, hashlib, importlib.util, argparse
import numpy as np, pandas as pd
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parent


def iaaft(x, rng, max_iter=200):
    x = np.asarray(x, float)
    values = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    for _ in range(max_iter):
        f = np.fft.rfft(y)
        phase = np.exp(1j * np.angle(f))
        z = np.fft.irfft(amp * phase, n=len(x))
        ranks = np.argsort(np.argsort(z, kind="stable"), kind="stable")
        new = values[ranks]
        if np.array_equal(new, y):
            break
        y = new
    error = float(
        np.linalg.norm(np.abs(np.fft.rfft(y)) - amp) / max(np.linalg.norm(amp), 1e-12)
    )
    return y, error


def asymmetry(x):
    return float(np.mean(np.diff(x) ** 3) / max(np.std(x) ** 3, 1e-12))


def longest_known(tokens, lookup):
    best = []
    cur = []
    for t in tokens:
        if t in lookup:
            cur.append(lookup[t])
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    return cur if len(cur) > len(best) else best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "seed": 42,
        "minimum_contiguous_known_tokens": 128,
        "maximum_tokens": 512,
        "maximum_documents": 40,
        "surrogates": 99,
        "iaaft_iterations": 200,
        "spectrum_error_limit": 0.05,
        "document_selection": "lexicographic SHA256 of raw document bytes; first eligible documents",
        "projection": "first PCA direction of normalized vocabulary vectors; no document-order fitting",
        "statistic": "standardized third moment of first differences",
        "correction": "Benjamini-Hochberg over all 40 planned documents; failed-quality tests receive p=1",
        "scope": "descriptive within-corpus diagnostic; fixed corpus-trained embeddings; not cross-language validation or authorship classification",
        "reference": "https://www.pks.mpg.de/tisean/TISEAN_2.1/docs/surropaper/node12.html",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    spec = importlib.util.spec_from_file_location(
        "prep", ROOT / "khasi_preprocessing_pipeline.py"
    )
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    with ZipFile(args.archive) as z:
        vocab = pd.read_csv(
            io.BytesIO(
                z.read(
                    "khasi_required_final/4_embeddings/cbow/khasi_cbow_word_vectors_dim32.csv"
                )
            )
        )
        vectors = vocab.iloc[:, 1:].to_numpy(float)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        lookup = {w: i for i, w in enumerate(vocab.token)}
        _, _, v = np.linalg.svd(vectors - vectors.mean(0), full_matrices=False)
        pc = v[0]
        projected = (vectors - vectors.mean(0)) @ pc
        np.save(out / "projection_pc1.npy", pc)
        candidates = []
        for name in z.namelist():
            if "/raw_texts_by_title/" not in name or not name.endswith(".txt"):
                continue
            raw = z.read(name)
            tokens = prep.preprocess_text_for_svd(raw.decode("utf-8-sig")).split()
            indices = longest_known(tokens, lookup)
            if len(indices) >= 128:
                candidates.append(
                    (hashlib.sha256(raw).hexdigest(), name, indices[:512])
                )
    candidates.sort()
    selected = candidates[:40]
    rng = np.random.default_rng(42)
    results = []
    for digest, name, indices in selected:
        arr = vectors[indices]
        x = projected[indices]
        x = (x - x.mean()) / max(x.std(), 1e-12)
        observed = asymmetry(x)
        null = []
        errors = []
        perm = []
        for _ in range(99):
            s, e = iaaft(x, rng)
            assert np.array_equal(np.sort(s), np.sort(x))
            null.append(abs(asymmetry(s)))
            errors.append(e)
            p = arr[rng.permutation(len(arr))]
            perm.append(float(np.einsum("ij,ij->i", p[:-1], p[1:]).mean()))
        cosine = float(np.einsum("ij,ij->i", arr[:-1], arr[1:]).mean())
        valid = max(errors) <= 0.05
        results.append(
            dict(
                document=name.rsplit("/", 1)[-1],
                sha256=digest,
                n_tokens=len(x),
                adjacent_cosine=cosine,
                permuted_cosine_mean=np.mean(perm),
                cosine_delta=cosine - np.mean(perm),
                time_asymmetry=observed,
                iaaft_p=(1 + sum(t >= abs(observed) for t in null)) / 100
                if valid
                else None,
                iaaft_max_spectrum_error=max(errors),
                surrogate_quality_pass=valid,
            )
        )
        print("document", len(results), "of", len(selected), flush=True)
    d = pd.DataFrame(results)
    valid = d.iaaft_p.notna()
    d["fdr_q"] = np.nan
    if valid.any():
        adjusted = false_discovery_control(d.iaaft_p.fillna(1.0).to_numpy())
        d.loc[valid, "fdr_q"] = adjusted[valid]
    d.to_csv(out / "document_results.csv", index=False)
    deltas = d.cosine_delta.to_numpy()
    boots = np.array(
        [rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(2000)]
    )
    summary = dict(
        eligible_documents=len(candidates),
        analysed_documents=len(d),
        surrogate_quality_pass=int(valid.sum()),
        fdr_rejections_at_005=int((d.fdr_q < 0.05).sum()),
        mean_cosine_delta=float(deltas.mean()),
        descriptive_document_bootstrap_ci=np.quantile(boots, [0.025, 0.975]).tolist(),
        limitations=[
            "Embedding training already rewards local co-occurrence; permutation effect is not independent evidence of language understanding.",
            "Documents from the same corpus/publisher need not be statistically independent.",
            "IAAFT rejection does not establish deterministic chaos or a causal dynamical model.",
            "No English/Russian or human/LLM comparison performed.",
        ],
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

# DIMER Mitra Regressor offline model package

Notebook Specification 1.0 requires an offline/DIMER model package to prove its model identity, immutable revision, and file integrity before model loading. The E2E tutorial therefore accepts DIMER ZIPs only in the manifested format below.

## Required archive layout

```text
dimer-model-manifest.json
model.safetensors
config.json
```

No additional files are permitted unless they are listed in the manifest.

## Manifest schema

```json
{
  "schema_version": 1,
  "model_id": "autogluon/mitra-regressor",
  "revision": "5f277aa8f69042d39d6ac3612aed18bb9279bd95",
  "files": [
    {
      "path": "model.safetensors",
      "size": 0,
      "sha256": "d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642"
    },
    {
      "path": "config.json",
      "size": 0,
      "sha256": "2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1"
    }
  ]
}
```

`size` must contain the actual byte size of each packaged file. The notebook verifies each declared size and digest, rejects missing or unlisted files, and then independently confirms that both files match the repository's pinned Mitra Regressor digests.

The archive may additionally be verified against a whole-ZIP SHA-256 supplied through `EXPECTED_DIMER_ZIP_SHA256` or the corresponding non-interactive environment input.

## Security boundary

The package reader rejects absolute paths, `..` traversal, backslash-based ambiguous paths, symlinks, oversized members, suspicious compression ratios, and excessive total expanded size. These checks protect extraction and integrity; they do not establish sender authenticity. Obtain a whole-archive digest through a trusted distribution channel when authenticity/substitution risk matters.

Legacy weight-only ZIPs intentionally fail this contract. Use the pinned-upstream path or rebuild the DIMER package with the v1 manifest rather than weakening verification.

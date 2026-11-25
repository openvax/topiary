# NetMHC-family stdout fixtures

Byte-faithful stdout captures from licensed DTU NetMHC-family binaries used by
`tests/test_cached_predictor.py` and `tests/test_cli_cached_predictor.py`.

## NetMHCcons captures (`netmhccons-a0101.stdout`, `netmhccons-b0801.stdout`)

Copied verbatim from `openvax/mhctools/tests/data/osteosarc/outputs/`
(provenance recorded in
`openvax/mhctools/tests/data/osteosarc/outputs/manifest.json`):

| File | Allele | Panel | SHA-256 |
|---|---|---|---|
| `netmhccons-a0101.stdout` | `HLA-A*01:01` | `cons_9mer` (17 9-mer peptides) | `f7a91d7cc7f6fa0b11a587cfc57f8461e68448743a6f08523e79464174a669e2` |
| `netmhccons-b0801.stdout` | `HLA-B*08:01` | `cons_9mer` (17 9-mer peptides) | `0462e3befe375655bf56674523fba482248e0d12abbaef9a6ea82eafcfa07a91` |

- **Commands**:
  - `netMHCcons -f {inputs}/cons_9mer.fasta -length 9 -a HLA-A01:01`
  - `netMHCcons -f {inputs}/cons_9mer.fasta -length 9 -a HLA-B08:01`
- **Licensed installations** (from `manifest.json`):
  - `netMHCcons-1.1` (`sha256: 486f26b9b4afaa3476f7e3227aae6de90ac087f359a571ed815761e57008734a`)
  - `netMHC-3.4` (`sha256: 203a9bbbba8c0ac437f03e82b598d7ee070d2620d2578a8a45259a5268316a31`)
  - `netMHCpan-2.8` (`sha256: 5e34ea9d15adccedb69dfb18e52800e384f5e2d618fb600b8427b90c3a253d71`)
  - `pickpocket-1.1` (`sha256: ec52d357a6f1c2402496a7b2a1a023bf53eaf6a66d9e7a8e612a47f981ba6c3e`)
- **Runtime**: container image `mhctools-test-netmhc-legacy:1`
  (`sha256:305827215f3867e5eb3b984d797e77beb3d96ad85d208dfeb8db3474396b4299`)

## Other fixtures (`netmhc_*`, `netmhcpan_*`, `netmhcstabpan_*`)

Captured from licensed DTU binaries (`netMHC 4.0`, `netMHCpan 4.0`,
`netMHCpan 4.1b`, `netMHCstabpan`) on peptide `SLLQHLIGL` across
`HLA-A*02:01`, `HLA-A*24:02`, and `HLA-B*07:02`.

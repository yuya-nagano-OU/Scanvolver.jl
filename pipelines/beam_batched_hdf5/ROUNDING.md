# beam分割で差が発生する場所の切り分け

2026-09-14、nside=256、lmax=200、bnmax=120、κ=20、Float32、2 skies、5 beams、tile幅4、8 ringの代表条件を追加調査した。

同じGPU上で同じWigner table・scan moment・K・sky Uを使い、全5 beamsと2+2+1 beamsを比較した。beamを分ける前後のprepared係数も完全一致を検査した。

- `apply_beams_gemm_gpu` の出力effective beam Eは、8 ringすべて・全tileで完全一致（最大差0）。
- 全5 beamsのEからビット単位で同じ列を抜き出し、同じUとともに `accumulate_convolution_tile_gemm_gpu!` に渡しても差が発生した。
- 差は最初のℓ tile（l=0:3）から発生。したがって、この条件では後段の蓄積行列積そのものが差を生むことを切り分けられた。

該当箇所は `src/convolution/contraction.jl:120` の `gemm_strided_batched!`。
pixelとscan weightごとに `C += U * E` を計算し、beam分割によって行列積の出力列数が5から2または1に変わる。縮約する `(spin,M,l)` 軸の長さ、数学的な式、入力値、ℓ tile順序は同じ。

| ring | Eの最大差 | 同一Eを与えたaccumulatorの最大差 |
|---:|---:|---:|
| 1 | 0 | 1.529947303e-5 |
| 2 | 0 | 2.673495086e-5 |
| 30 | 0 | 2.388360178e-5 |
| 256 | 0 | 2.289648834e-5 |
| 512 | 0 | 2.102329753e-5 |
| 768 | 0 | 2.479752419e-5 |
| 1022 | 0 | 3.433279314e-5 |
| 1023 | 0 | 2.198176289e-5 |

表は複素accumulatorの差であり、最終I/Q/Uの差とは別の量。各ringで全tileとbeam batchにわたる最大値を集計した。

この結果は「どのAPI呼出しから数値差が発生するか」を確認したもの。cuBLAS内部で選択された具体的なkernelや加算木までは追跡していないため、その詳細は断定しない。HDF5の保存前後は別の比較でビット一致を確認済み。

再現（リポジトリのルートで実行）：

```sh
EBCODE_INPUT_ROOT=/path/to/inputs-root julia --project=pipelines/beam_batched_hdf5 --threads=4 pipelines/beam_batched_hdf5/test/rounding_probe.jl
```

実測データは検証したcheckoutの `pipelines/beam_batched_hdf5/test-output/rounding_probe_8050532937901272.toml`。

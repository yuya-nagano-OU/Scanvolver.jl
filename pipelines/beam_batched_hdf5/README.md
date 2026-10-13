# Beam分割・ring保存pipeline

EBCoDeCMBの公開APIを組み合わせ、beamをbatch処理しながら完成したringをHDF5へ保存する。パッケージ本体の変更は不要。新しい入口は `convolve_to_hdf5` で、全空mapを作らず出力ファイルの絶対パスを返す。

置き場所はリポジトリの `pipelines/beam_batched_hdf5`。

## 実行環境と比較テスト

RTX A6000 2枚で検証済み。比較テストはJuliaを4 threadsで起動する。pipeline自体は1 GPUでも使える。専用Projectが相対パスでこのcheckoutのEBCoDeCMBを参照するため、rootのProject/Manifestを変更しない。

```sh
# リポジトリのルートで実行する
EBCODE_JULIA=julia     # Julia 1.12
EBCODE_PYTHON=python   # h5py が入った Python
export EBCODE_INPUT_ROOT=/path/to/inputs-root   # representative.jl が読む入力の置き場所
"$EBCODE_JULIA" --project=pipelines/beam_batched_hdf5 -e 'using Pkg; Pkg.instantiate()'
mkdir -p pipelines/beam_batched_hdf5/test-output

EBCODE_SMALL_OUTPUT=$(mktemp -d "$PWD/pipelines/beam_batched_hdf5/test-output/recheck_small_XXXXXX")
EBCODE_PIPELINE_TEST_OUTPUT="$EBCODE_SMALL_OUTPUT" "$EBCODE_JULIA" --project=pipelines/beam_batched_hdf5 --threads=4 pipelines/beam_batched_hdf5/test/test_beam_batched_pipeline.jl
"$EBCODE_PYTHON" pipelines/beam_batched_hdf5/test/verify_h5py.py "$EBCODE_SMALL_OUTPUT"

EBCODE_REP_OUTPUT=$(mktemp -d "$PWD/pipelines/beam_batched_hdf5/test-output/recheck_representative_XXXXXX")
EBCODE_PIPELINE_TEST_OUTPUT="$EBCODE_REP_OUTPUT" "$EBCODE_JULIA" --project=pipelines/beam_batched_hdf5 --threads=4 pipelines/beam_batched_hdf5/test/representative.jl
"$EBCODE_PYTHON" pipelines/beam_batched_hdf5/test/verify_h5py.py "$EBCODE_REP_OUTPUT"
```

`test_beam_batched_pipeline.jl` は入力を内部生成し、nside=8、2 skies、5 beamsをbatch 1/2/5、Float32/Float64、1/2 GPUで比較する。`representative.jl` はCMB seed 0/1と5個の回転Gaussian beamを読み、nside=256、lmax=200、bnmax=120、κ=20の8 ringで比較する。走査点はpixelの角付近に生成する小さなfixtureであり、実際の年間走査ではない。入力の置き場所は環境変数 `EBCODE_INPUT_ROOT` で指定する（必須）。

出力は `metrics.toml`、HDF5 map、参照mapの `.npy`、Python検証の `h5py_verification.json`。参照との比較に限り、テスト側では小規模の全空mapをRAMに読み込む。pipelineにはこの割当てはない。`test-output/` と専用 `Manifest.toml` はgitignore対象で、検証時のManifestは検証したマシンに残す。

## 呼び出し方

既存APIで用意した `cp::ConvolutionParams`、`buffer::RingScanBuffer`、同じ精度の `PreparedAlm` / `PreparedBlm` を渡す。

```julia
using EBCoDeCMB
include(joinpath(pkgdir(EBCoDeCMB), "pipelines", "beam_batched_hdf5", "beam_batched_pipeline.jl"))
using .BeamBatchedHDF5

# cp, buffer, raw_alms, raw_blmsは利用する入力から準備する。
alm = prepare_alm(cp, raw_alms; precision=Float32)
blm = prepare_blm(cp, raw_blms; precision=Float32)
path = convolve_to_hdf5(cp, buffer, alm, blm;
    output_path="/absolute/existing/output-directory/map.h5",
    beam_batch_size=2,
    devices=[0, 1],
    beta_order=8,
    l_tile=:auto,
    max_l_tile=32,
    sky_ids=["sky-$i" for i in 1:cp.numofsky],
    beam_ids=["beam-$i" for i in 1:cp.numofbeam],
    metadata=Dict("scan_id" => "identifier-of-input-scan"),
    progress_callback=r -> println("saved ring ", r.ring),
)
ring_map = read_ring_hdf5(path, 1) # (sky, beam, I/Q/U, ring_pixel)
subset = read_ring_hdf5(path, 1; skies=1:1, beams=1:2)
```

出力先の親directoryは事前に作る。既存ファイルへの上書きは拒否する。`rings` を省略するとbufferに含まれる全ringが対象で、整数のvector/rangeで一部ringを指定できる。`beam_metadata` にはbeam数と同じ長さのvectorを渡せる（例えばtheta/phi/psiや入力ファイル名）。`metadata` はHDF5で保存可能な値の辞書。IDの順序はprepared入力の順序と一致させる。

compute精度はprepared入力のFloat32/Float64から決定し、`map_precision` の既定値も同じ精度。`rcond=1e-2`、`unseen=NaN`、`moment_method=:gemm` が既定値。`ring_stokes` はI/Q/Uを返す。未観測pixel、未要求ring、条件の悪い観測行列で解けないpixelは `unseen` になる。HWP/noiseはこの入口では受け付けない。

## βMAXと再利用

βMAXは **出力nside** の `Healpix.max_pixrad(Resolution(nside))` に、Float64の回転計算とcompute精度への変換の余裕を加えて上向きに丸める。

```text
radius = max_pixrad(Resolution(nside_output))
margin = 64*eps(Float64) + 8*eps(compute_precision)*radius
beta_max = nextfloat(compute_precision(radius + margin))
```

Float32ではnside=256で約0.004172565 rad、nside=512で約0.002087058 rad。同じ `ScanMomentPlan` をWigner table構築とscan moment構築に渡す。各ringの実測βを、momentを作る前に有限値かつ `[0,beta_max]` 内と検査する。逸脱は例外で停止し、外挿しない。βMAXを自動化してもChebyshev次数の精度保証まで自動化するものではなく、`beta_order` は計算条件として明示的に保存する。

処理順序は次のとおり。

1. GPUごとに全skyのalm、Wigner table、ringD contextを保持する。
2. ringごとにangles、scan moment、ringDを一度作り、beam全体のring accumulatorをbatchごとに確保する。
3. ℓ tileごとにbeam非依存kernel Kとsky FFT Uを一度作る。同じK/Uに各beam batchを適用してaccumulateする。
4. 全ℓのaccumulate後、batchごとにringのI/Q/Uを求めてHDF5へ書き、全batchの保存後にring完了を記録する。

beam係数は既にprepare済みのsliceを分ける。`prepare_blm` を再度適用して二重に共役化することはない。公開 `build_sky_ring_gpu` を使用する。shape固有のFFT planはring内の全ℓ tileで再利用し、ring終了時にcuFFTのidle handleごと明示破棄する。polar ringごとに異なるFFT長のworkspaceを蓄積しない。

GPU上に全beam分残すのは現在のringのaccumulatorであり、その大きさは `sizeof(Complex{T}) * num_sky * num_beam * active_pixels * 2` bytes。大きいbeam中間配列はbatchサイズに制限する。CPU側では入力、scan buffer、beamのprepared copyに加え、現在のringのbatch mapを保持する。sky自体の分割はしない。

`l_tile=:auto` は、Kと全ring Uとactive Uを同時に保持する寿命を考慮した見積りから幅を選ぶ。moment/ringD/accumulator確保後の利用可能GPUメモリに対し、既定で70%を上限にする。cuFFT workspaceはplan依存なので明示バッファ見積りに含めず余裕を残す。利用可能量が1 multipoleにも足りなければ明示的に失敗する。これは本番規模での容量測定を置き換えるものではない。

## HDF5 v1形式

| 要素 | 意味 |
|---|---|
| `map` | disk/h5pyのshapeは `(sky, beam, 3, pixel)`、Stokes順序I/Q/U、HEALPix RING順序 |
| `sky_ids`, `beam_ids` | 各軸に対応する入力ID |
| `beam_metadata/` | 各beamに対応する追加属性vector |
| `metadata/` | 利用者が指定した入力・走査などの条件 |
| root attributes | schema、精度、nside、ℓ範囲、bnmax、κ、βMAX/次数、batch/tile設定、rcond、unseenなど |
| `rings/pixel_first`, `rings/pixel_last` | ringのグローバルpixel範囲。**1始まり・両端を含む** |
| `rings/requested`, `rings/completed` | 要求されたringと保存が完了したringの0/1 flag |
| `rings/device`, `rings/observed_beta_max` | 使用GPUと実測βの最大値 |
| `rings/elapsed_seconds` | angles生成からmap書込みと最初のflushまでのring実時間。context準備・最後の完了記録flushは含まない |
| `rings/*_builds`, `rings/tile_count`, `rings/beam_applications` | 再利用回数の検証用記録 |
| `rings/estimated_tile_bytes`, `rings/accumulator_bytes` | 明示GPUバッファの見積り。実測peak使用量ではない |
| `run_state` | 0=実行中、1=要求ringがすべて完了、2=失敗 |

HDF5.jlは次元を反転してdiskへ保存するため、Juliaで `file["map"]` を直接読むと `(pixel, stokes, beam, sky)`。`read_ring_hdf5` はEBCoDeCMBと同じ `(sky, beam, stokes, ring_pixel)` へ戻す。固定長の文字列属性はh5pyではbytesとして返る。

```python
import h5py
with h5py.File("/absolute/path/map.h5", "r") as f:
    ring = 1  # この例ではring番号も1始まり
    assert f["rings/completed"][ring - 1] == 1
    first = int(f["rings/pixel_first"][ring - 1])
    last = int(f["rings/pixel_last"][ring - 1])
    ring_map = f["map"][:, :, :, first - 1:last]
```

mapはchunked datasetで、ringのhyperslabを順次書く。HDF5アクセスとcallbackは共通lockで直列化し、GPU workerをすべてjoinしてからファイルを閉じる。各ringのmapを書いてflushした後にcompleted flagを書いてflushする。例外時はGPU資源をfinallyで解放し、完了したringの記録を残す。OS強制終了・電源断時のトランザクション保証や自動再開は今回の実装には含まない。ファイルの読出しは処理終了後に行い、部分ファイルでは必ずcompletedを確認する。

`batch_callback` は書込み後・ring完了前にring/batchだけの借用CPU配列を受け取る。`progress_callback` は完了記録後に呼ぶ。callback例外も呼出し元に伝播する。

検証結果、受入れ誤差、対象外の条件は同directoryの `VALIDATION.md` に記録した。

# 実装検証結果

2026-09-13〜14 JST、RTX A6000 2枚のマシンで実装と小規模比較を完了した。
対象は EBCoDeCMB.jl の
branch `chore/profiling-split`、HEAD `e7e721b8218a691171f259ae160b16ddbb8c48b3`。

## 変更範囲

追加したファイルは `pipelines/beam_batched_hdf5/` 以下の8ファイルのみ。

- `beam_batched_pipeline.jl`: βMAX自動化、beam batch、ring処理、GPU管理、HDF5 writer/reader。
- `Project.toml`, `.gitignore`: 独立実行環境と生成物の除外。
- `test/test_beam_batched_pipeline.jl`: 自己完結したFloat32/Float64の既存driver比較、異常系と再実行。
- `test/representative.jl`: 既存CMB/回転beam入力でnside=256の比較。
- `test/verify_h5py.py`: Pythonによるdisk軸順序、pixel位置、数値の独立読出し検証。
- `README.md`, `VALIDATION.md`: 利用・再現手順と結果。

パッケージ本体、rootのProject/Manifestは変更していない。以前から未追跡だった
`notebook/one_pixel_persistent_scan_moments_comparison.ipynb` と `script/A6000_double/` を変更せず保持した。commit/pushは行っていない。

## 環境と実行結果

Julia 1.12.6、CUDA.jl 6.3.1、HDF5.jl 0.17.3、Healpix.jl 4.2.4、NPZ.jl 0.4.3、
RTX A6000 2枚。Python側はh5py 3.10.0、NumPy 1.24.4。
解決済み依存環境は `pipelines/beam_batched_hdf5/Manifest.toml` に残した（gitignore対象）。
再現コマンドは `README.md` を参照。

| 検証 | 結果 |
|---|---|
| nside由来βMAX、境界点、Float32/64 | 59/59 pass |
| nside=8、2 skies×5 beams、batch 1/2/5、1/2 GPU、異常系 | 538/538 pass |
| nside=256、既存2 skies×5 beams、8 ring、batch 2、2 GPU | 50/50 pass |
| h5pyによる全比較出力の読出し | 9/9ファイル pass |

Juliaの合計は **647 assertionがpass**。小規模suiteは49.8秒、代表条件suiteは36.4秒（コンパイル等を含む実行時表示）。後者は既存driverが20.62秒、新pipelineが7.75秒だったが、起動後のコンパイル状態と空ring処理が異なるため、この時間から速度向上率は評価しない。

保存先（すべて同checkoutの `pipelines/beam_batched_hdf5/test-output/` 以下）：

- `run_8004926153850835/`: 最終小規模suiteの `metrics.toml`、8比較HDF5、参照 `.npy`、異常系部分出力、`h5py_verification.json`。
- `representative_8043778981535300/`: 代表条件の `metrics.toml`、比較HDF5、参照 `.npy`、`h5py_verification.json`。metricsには7入力ファイルの絶対パスとSHA256も記録。

## 比較条件と受入れ基準

比較相手は現在のEBCoDeCMBの既存 `convolve_sphere_gpu` / `convolve_sphere_multigpu`。
旧 `EBCoDe_gpu_recursion` の計算結果との比較ではない。代表条件では旧directoryにあるbeam**入力ファイルだけ**を読む。

両実装に同じprepared alm/blm、scan buffer、Wigner/scan plan、compute/map精度、ℓ tile幅、rcond=0.01、moment GEMMを使用した。β次数8、tile幅4（最終tileに端数あり）。基準は新pipelineの結果を見る前にテスト内に固定し、結果に合わせて緩めていない。

有効値の各要素に対し `abs(new-reference) <= atol + rtol*abs(reference)` を要求。

| fixture | atol | rtol |
|---|---|---|
| 正規化した小規模入力、Float64 | 1e-12 | 1e-11 |
| 正規化した小規模入力、Float32 | 2e-6 | 2e-5 |
| 代表条件、Float32 | 2e-6 × 参照Stokesごとの最大振幅 | 2e-5 |

以下の相対誤差は有効値の `norm(new-reference)/norm(reference)`。点ごとの判定は上の混合誤差で行い、ゼロ付近での相対誤差の発散を避けている。最大絶対誤差は入力と同じ単位。

## 小規模入力の数値差

nside=8、lmax=12、bnmax=4、κ=3。非軸対称beamと異なるskyの物理的な実数場のpacked T/E/B係数を内部生成した。12 ringに観測を配置し、疎なringと全pixelにhitのあるring、未観測ring、偏光角不足で解けないpixelを含めた。

表はbatch 1/2/5および1/2 GPUの比較で、各Stokesの最も大きい値。

| 精度 | Stokes | 最大絶対誤差 | 相対L2誤差 |
|---|---|---:|---:|
| Float64 | I | 1.776357e-15 | 4.894899e-16 |
| Float64 | Q | 1.110223e-16 | 3.853773e-16 |
| Float64 | U | 1.110223e-16 | 4.111504e-16 |
| Float32 | I | 1.072884e-6 | 2.593459e-7 |
| Float32 | Q | 5.960464e-8 | 2.079151e-7 |
| Float32 | U | 6.705523e-8 | 2.054062e-7 |

batch=2は **2+2+1** の端数を含む。batch=5は両精度で有効値の差がゼロ。
batch=2の1 GPUと2 GPUの比較誤差も同じだった。
最大の `error/(atol+rtol*abs(reference))` はFloat64で0.00016014、Float32で0.050527（合格上限1）。

## nside=256の代表条件

既存CMB seed 0/1の2 skyと、FWHM=5 deg、theta=10/20/30/40/50 deg、phi=psi=0の5 beam。
beam生成nside=512、出力nside=256、lmax=200、bnmax=120、κ=20。
skyはpacked配列をlmax=200に外部で切り出して両実装に同一入力として渡した。

8 ring `[1,2,30,256,512,768,1022,1023]`、計88 pixelの各pixelに8方向の観測を与えた。scanは合成した角付近のpointingであり、実走査データではない。参照の有効値は各Stokes880個（88 pixel×2 sky×5 beam）。

自動βMAXは **0.004172564949840307 rad**、実測最大βは **0.004151697993246309 rad**。
従来の固定値0.004を超えるpointingを含み、自動範囲内で計算できた。

| Stokes | 最大絶対誤差 | 相対L2誤差 | 最大許容誤差比 |
|---|---:|---:|---:|
| I | 3.242493e-5 | 1.584120e-7 | 0.070434 |
| Q | 3.433228e-5 | 2.017170e-7 | 0.154218 |
| U | 3.814697e-5 | 1.741900e-7 | 0.091018 |

絶対許容誤差はI=1.934636e-4、Q=1.677775e-4、U=1.441173e-4に、各点の相対許容誤差を加えた値。すべて事前の基準内。

8 ringすべてでmoment/ringDは各1回、K/sky FFTは各51回、beam適用は各153回（51 tile×3 batch）。2 GPUの両方がringを担当した。これによりbeamごとにsky FFTやKを再生成していないことも確認した。

## 保存・異常系の確認

- 保存直前の各ring/batchのCPU配列のSHA256と、JuliaでHDF5から読み戻した配列のSHA256が一致した。保存・読み戻しによる値の変化はない。
- h5pyで軸の並べ替えなしに参照 `.npy` と比較した。shapeは小規模 `(2,5,3,768)`、代表条件 `(2,5,3,786432)`。sky/beam順序と全pixel位置を含む比較が通った。
- ID、beam metadata、ringの先頭/末尾pixel、completed/requestedを検査。NaN/有限値maskは全比較で既存driverと完全一致した。
- 故意にwriter callbackを失敗させ、β範囲外入力も与えた。対応する原因の例外が伝播し、ファイルはrun_state=2、失敗ringはcompleted=0になった。
- その後同じJulia processで2 GPUの計算を再実行できた。未要求ringは未完了、要求した空ringは完了かつNaNという扱いも確認した。
- 最終確認時、両GPUは使用率0%、memory.usedは各10 MiBで、テストprocessは終了していた。

開発中に修正した失敗はCUDA/HDF5の同名 `attributes` の解決、テスト出力pathの正規化、h5pyのbytes文字列のdecodeだった。数値許容誤差の変更による通過は行っていない。

今回確認したのは既存の数値計算を新しい分割・保存順序で再現すること。独立した物理oracleに対する精度、本番メモリ使用量、年間scan、100 skies×288 beamsの本番計算、自動再開は対象外。この小規模検証の完了時点で停止した。

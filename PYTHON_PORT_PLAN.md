# VTEA Python Port — Implementation Plan

## Goal and scope

Fully replace the Java/ImageJ1/SciJava VTEA application with a Python-native equivalent, distributed as a **napari plugin** backed by a standalone, headless-usable **core analysis library**. Java is retired once parity is reached; it is not kept as a permanent hybrid host (the codebase already tried that pattern for deep learning — see "Why not extend the existing Py4J bridge" below).

**Primary motivation:** several of VTEA's Java dependencies are effectively unmaintained or awkward wrappers around ecosystems that are Python-native and actively developed — JavaCPP/PyTorch bindings for the 3D VAE/CNN stack, a subprocess+socket (Py4J) bridge to run Cellpose (itself a Python package), Renjin embedding an old R interpreter for a handful of plotting calls, and JFreeChart/XChart/vioplot in place of matplotlib/plotly. Porting removes these translation layers instead of adding more of them.

## Current state (facts, from codebase inventory)

- **451 Java files, ~109K LOC**, Maven build, Java 8, ~30 packages under `src/main/java/vtea/*`, `vteaobjects`, `vteaexploration`.
- **Entry point:** `vtea._vtea` — legacy ImageJ1 `PlugIn` (its `@Plugin` SciJava annotation is commented out, so it is *not* SciJava-discovered; ImageJ1's `run()` is the real bootstrap). Registered via `src/main/resources/plugins.config`.
- **Plugin/extension architecture:** 13 `vtea.services` classes, each binding a SciJava `PluginService` lookup to an extension-point interface (`Segmentation`, `FeatureProcessing`, `Measurements`, `Morphology`, `LUT`, `GateMath`, `PlotMaker`, `Processor`, `Workflow`, `FileType`, `ImageProcessing`, `NeighborhoodMeasurements`). This is the registry that populates every algorithm dropdown in the UI.
- **Largest packages:** `vtea.objects` (51 files, ~15.5K LOC — segmentation engine, ~15 methods including LayerCake3D/kD-tree, FloodFill3D, MorphoLibJ/ImgLib2 connected components, Cellpose/DeepImageJ, STAPLE), `vtea.protocol` (63 files, ~15.5K LOC — the block-based visual pipeline builder), `vtea.deeplearning` (42 files, ~15.5K LOC — two *parallel* deep-learning integrations, see below), `vtea.exploration` + `vteaexploration` (116 files, ~25K LOC combined — the interactive plotting/gating workbench, `MicroExplorer` main window).
- **Deep learning has two independent paths today:** (1) a Py4J bridge (`CellposeInterface`) that launches `python/cellpose_server.py` as a subprocess and calls it over a `py4j.GatewayServer` socket, with its own restart/backoff and GPU-OOM detection logic; (2) a from-scratch Java 3D VAE/CNN stack on JavaCPP `pytorch-platform`/`cuda-platform` bindings, with its own Swing training UI. Neither talks to the other.
- **Persistence:** `vtea.jdbc.H2DatabaseEngine` — in-memory H2 (`jdbc:h2:mem:VTEADB`), two tables (`MEASUREMENTS`, `OBJECTS`) as the session-scoped store for per-object features. Not disk-persisted by default.
- **R/Renjin:** minimal — `vtea.renjin` (149 LOC) only generates an R color-palette string; despite `ggplot2`/`gplots`/`vioplot` CRAN deps in `pom.xml`, no substantial R usage was found elsewhere.
- **Large-volume support (VTEA 2.0, in progress):** `vtea.dataset.volume` (`VolumeDataset`/`ImagePlusVolumeDataset`/`ZarrVolumeDataset`), `vtea.io.zarr`, `vtea.partition` (`Chunk`, `VolumePartitioner`, `ChunkIterator`, `ObjectStitcher` using a Smile kD-tree for boundary object merging), `vtea.objects.AbstractChunkedSegmentation`. Per the source doc's own "Remaining Work" section, this is partially implemented, not finished — treat as aspirational.
- Two sample datasets already sit at repo root (`AQtest_human_crop.tif`, `C1-IU_VTEA_ExampleData_001.tif`) — useful as the seed for a parity-test golden dataset.

### Why not just extend the existing Py4J bridge (hybrid path)

That's the architecture already used for Cellpose, and the codebase shows its cost first-hand: a subprocess launcher, a socket gateway, JSON parameter marshaling, manual byte-array (de)serialization of images in both directions, and bespoke restart/OOM-recovery logic — all to call a library that's a native `pip install` away. Multiplying that pattern across segmentation, clustering, DR, and plotting would mean permanently maintaining two runtimes, two dependency trees, and a marshaling layer between them, for a codebase whose GUI, algorithms, and glue code all need substantial rework regardless. A full port removes the bridge instead of scaling it.

## Target architecture

Two packages:

- **`vtea-core`** — pure Python library, no GUI dependency. Data model, I/O, segmentation, features, clustering/DR, gating, deep learning. Usable headless from scripts/Jupyter/CLI/HPC.
- **`vtea-napari`** — napari plugin (dock widgets + `npe2` manifest) that is a thin UI layer over `vtea-core`. napari is Qt-based (PyQt5/PySide2) and is the closest Python analog to the ImageJ/Fiji viewer VTEA plugs into today, with an active plugin ecosystem and native 3D volume rendering.

### Dependency mapping

| Java (today) | Python (target) | Notes |
|---|---|---|
| ImageJ1/ImageJ2, SciJava plugin framework | napari + `npe2`/`stevedore`/entry-points registry | Extension points (segmentation, clustering, DR, morphology, LUT, plot makers, workflows) become entry-point groups, same role as the 13 `vtea.services` classes |
| ImgLib2 (n-dim images) | NumPy + Dask arrays, `xarray` for labeled axes | |
| N5 / Zarr, `vtea.io.zarr`, `vtea.partition` | `zarr-python`, `dask.array` (`map_blocks`/`map_overlap`) | Dask's built-in chunking/overlap replaces most of the hand-written `Chunk`/`VolumePartitioner`/`ChunkIterator`; only `ObjectStitcher`'s cross-chunk object-merge logic needs a genuine port |
| Bio-Formats / OME-TIFF import | `bioio` (or `aicsimageio`) for proprietary formats, `tifffile` for TIFF/OME-TIFF, `ome-zarr-py` | `bioio` still uses a JVM under the hood for exotic vendor formats (Zeiss CZI, Leica LIF, etc.) via `scyjava` — that's a transparent runtime dependency, not maintained application code |
| MorphoLibJ, ImgLib2 connected components | `scikit-image` (`morphology`, `segmentation.watershed`, `measure.label`), `scipy.ndimage` | |
| Smile (KMeans/GMM/hierarchical/kD-tree), la4j | `scikit-learn` (`KMeans`, `GaussianMixture`, `AgglomerativeClustering`), `scipy.spatial.cKDTree` | X-Means/G-Means/deterministic annealing have no direct sklearn equivalent — port the BIC/AIC model-selection logic directly |
| `tsne` library, Isomap, Laplacian Eigenmap | `scikit-learn` (`TSNE` or `openTSNE`, `Isomap`, `SpectralEmbedding`), `umap-learn` (new option) | sklearn already ships Isomap and spectral embedding built-in |
| JFreeChart, XChart, vioplot | `matplotlib`/`seaborn` (violin plots), embedded in Qt dock widgets; `plotly` optional for interactive | |
| Swing (`MicroExplorer`, `ProtocolManagerMulti`, gate manager, morphology dialogs, plot windows) | `napari` dock widgets, `magicgui`, raw `qtpy` (PyQt5/PySide2) | See "Highest-risk area" below re: the protocol builder |
| H2 (in-memory JDBC) | `DuckDB` (embedded, columnar, SQL, native Arrow/pandas interop) | Backs the `MEASUREMENTS`/`OBJECTS` tables; also enables on-disk persistence if wanted later |
| Renjin/R (color palette only) | `matplotlib`/`seaborn` colormaps | Drop the R dependency entirely — usage found is a single palette string |
| JavaCPP PyTorch bindings (3D VAE/CNN) | native `torch`, optionally `MONAI` for ready-made 3D medical-imaging architectures | Removes an entire binding layer; direct access to `torch.compile`, mixed precision, model zoo |
| Py4J bridge + `python/cellpose_server.py` subprocess | in-process `cellpose` import | Deletes `CellposeInterface`, the subprocess/socket plumbing, and the bridge script entirely |
| "DeepImageJ" generic model inference | `bioimageio.core` (the actual current successor to DeepImageJ, same BioImage Model Zoo spec) | |
| JNI stub (`HelloJNI`, unused) | — | Drop, not functionally wired in today |

## Highest-risk area: the protocol builder

`vtea.protocol` is 63 files / ~15.5K LOC of bespoke drag-and-drop pipeline-building Swing UI — the single largest, hardest-to-mechanically-port piece in the codebase, and it doesn't map cleanly onto any existing Python widget toolkit. Before Phase 4 starts, make an explicit scope call rather than defaulting to a pixel-for-pixel Swing clone:

- **Option A — full visual clone:** custom Qt node/block editor replicating today's drag-and-drop pipeline builder. Highest fidelity, highest UI cost.
- **Option B — scriptable pipeline + light step-list UI (recommended default):** the pipeline becomes a plain Python object (ordered list of steps, each backed by the entry-point registry), scriptable directly from Jupyter/CLI, with a simpler linear step-list/parameter-form UI in napari rather than a full drag-and-drop canvas. Saved Java workflow XML gets an import converter into this format so existing user pipelines aren't stranded.

Recommend confirming with actual VTEA users which they rely on before committing engineering weeks here — this is the one area where scope, not technique, drives the estimate.

## Phased roadmap

| Phase | Content | Est. effort |
|---|---|---|
| **0. Foundations & parity harness** | `vtea-core`/`vtea-napari` package skeletons, CI, dependency choices locked in. Build a golden-dataset regression harness: run today's Java pipeline against the two sample TIFFs (already in-repo) and any other representative datasets, capture segmentation masks, feature tables, and cluster/DR outputs as fixtures for every later phase to diff against. | 2–3 weeks |
| **1. Core data model & I/O** | `VolumeDataset` (NumPy in-memory + Dask/Zarr chunked) replacing `vtea.dataset`+`vtea.partition`+`vtea.io.zarr`; object model as labeled arrays + DuckDB/pandas measurement tables replacing `vteaobjects.MicroObject`+H2; readers via `bioio`/`tifffile`/`ome-zarr-py`. | 3–4 weeks |
| **2. Algorithm core** (largest phase) | Segmentation (~15 methods), feature/measurement extraction (`regionprops_table`-based), clustering (KMeans/GMM/hierarchical via sklearn; X-Means/G-Means/annealing ported directly with their BIC/AIC logic), DR (PCA/t-SNE/Isomap/Laplacian Eigenmap), gating, spatial stats, image preprocessing. | 6–10 weeks |
| **3. Deep learning consolidation** | Native PyTorch (optionally MONAI) replacing the JavaCPP VAE/CNN stack; in-process `cellpose` replacing the Py4J bridge; `bioimageio.core` replacing DeepImageJ integration. | 3–5 weeks |
| **4. napari plugin (GUI)** | Dock widgets recreating `MicroExplorer` (plots via matplotlib/vispy), gate manager (`PolygonSelector`/`LassoSelector` or vispy-based interactive gating), pipeline builder per the scope decision above, LUTs (largely free via napari's built-in colormaps), heatmap/violin plot widgets. | 6–10 weeks, pending Option A/B decision |
| **5. Parity validation & cutover** | Run the Phase 0 golden-dataset suite end-to-end: segmentation IoU, feature-table numeric diffs, cluster-assignment ARI, against Java outputs. Beta with real users. Docs + workflow-XML migration converter. | 2–4 weeks |
| **6. Decommission Java** | Archive/tag the Java codebase, update the Fiji update site listing to point at the new pip/conda package and napari plugin index. | 1 week |

**Total: roughly 23–37 engineer-weeks (~6–9 months at one senior engineer, less with parallelism across phases 2–4 once Phase 1 lands).**

## Other risks / open questions

- **Numerical parity for custom algorithms** (X-Means, G-Means, deterministic annealing, kD-tree boundary stitching) — these have no drop-in library replacement and must be ported logic-for-logic, then validated against the Phase 0 fixtures.
- **Chunked large-volume behavior** — Dask's overlap/stitching semantics differ from the hand-rolled `Chunk`/`ObjectStitcher` system; validate object counts/boundaries match on a volume too large for RAM before trusting it in production.
- **ImageJ macro compatibility** — `vtea.imageprocessing.builtin.IJMacro` lets users embed arbitrary ImageJ1 macros in a pipeline step. Decide explicitly: drop macro support (replace with scikit-image equivalents), or keep an optional `pyimagej` bridge for legacy macro compatibility during the transition window.
- **Format coverage** — confirm `bioio`'s JVM-backed readers cover every vendor format current users rely on before dropping Bio-Formats from the primary code path.
- **User pipeline continuity** — saved `.xml` workflow/protocol files from the Java app should have a conversion path into the new format (called out in Phase 5) so existing collaborators aren't blocked mid-migration.

## Suggested next step

Confirm the Phase 4 protocol-builder scope (Option A vs B) with VTEA's actual users, since it's the single largest swing factor in the GUI estimate, then start Phase 0 (parity harness) — it's a prerequisite for validating every later phase and has no dependency on the scope decision.

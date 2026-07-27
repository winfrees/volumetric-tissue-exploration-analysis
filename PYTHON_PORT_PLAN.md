# VTEA Python Port — Implementation Plan

> **Status: the port exists and is well underway — it lives at
> [winfrees/vtea-python](https://github.com/winfrees/vtea-python).**
> Phases 0–4 below are complete (core library, algorithms, deep learning,
> and the napari GUI), plus a standalone Linux/Windows runtime published
> automatically to GitHub Releases. Phase 5 (parity validation) is the
> current front, and it's the phase *this* repo feeds: the
> `GoldenFixtureGenerator` and the "Generate golden-dataset fixtures"
> workflow here produce the Java reference outputs the Python port diffs
> against.
>
> This document is kept as the Java-side view of the port — the original
> inventory and dependency analysis it was planned from, updated with what
> the implementation actually found. The **living** plan, which tracks
> module-by-module status, is
> [`docs/PORT_PLAN.md`](https://github.com/winfrees/vtea-python/blob/main/docs/PORT_PLAN.md)
> in the port repo. Where the two disagree, that one is current.
>
> See "What the implementation changed" below for the places where
> building it corrected the assumptions made here.

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

- **`vtea-core`** — pure Python library, no GUI dependency. Data model, I/O, segmentation, features, clustering/DR, gating, classification, and the headless `Step`/`Pipeline` workflow engine. Usable headless from scripts/Jupyter/CLI/HPC. (There is deliberately no separate `deeplearning` subpackage — see "What the implementation changed".)
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
| Swing (`MicroExplorer`, `ProtocolManagerMulti`, gate manager, morphology dialogs, plot windows) | `napari` dock widgets, raw `qtpy` (PyQt5/PySide2), `matplotlib` for the embedded scatter plot | Shipped without `magicgui` — see the protocol-builder section below for why. |
| H2 (in-memory JDBC) | `DuckDB` (embedded, columnar, SQL, native Arrow/pandas interop) | Backs the `MEASUREMENTS`/`OBJECTS` tables; also enables on-disk persistence if wanted later |
| Renjin/R (color palette only) | `matplotlib`/`seaborn` colormaps | Drop the R dependency entirely — usage found is a single palette string |
| JavaCPP PyTorch bindings (3D VAE/CNN) | native `torch`, in `vtea_core.classification` | Removes an entire binding layer. Shipped without MONAI — a plain `nn.Module` 3D CNN covered it. Optional `deeplearning` extra, so a base install doesn't pull multi-GB PyTorch |
| Py4J bridge + `python/cellpose_server.py` subprocess | in-process `cellpose` import, in `vtea_core.segmentation` | Deletes `CellposeInterface`, the subprocess/socket plumbing, and the bridge script entirely |
| "DeepImageJ" generic model inference | `bioimageio.core` (the actual current successor to DeepImageJ, same BioImage Model Zoo spec) | **Deferred** — the one Phase 3 item not yet ported; more involved than Cellpose and no current user depends on it |
| JNI stub (`HelloJNI`, unused) | — | Drop, not functionally wired in today |

## The protocol builder — resolved: Option A, and cheaper than feared

This was flagged here as the highest-risk area, on the assumption that `vtea.protocol` (63 files / ~15.5K LOC) was a bespoke **drag-and-drop** pipeline canvas that wouldn't map onto any Python widget toolkit. Reading the actual `ProtocolManagerMulti`/`blockstepgui` source before porting corrected that: **there is no drag-and-drop.** A `grep` for `TransferHandler`/`DragSource`/`DropTarget`/drag-gesture code across the whole package turns up nothing — the layout is a plain `FlowLayout`, there is no wire-based connection UI and no drag-to-reorder. The protocol is an ordered, numbered stack of step cards (process name, parameter-summary comment, thumbnail preview, Edit/Delete) built by adding steps from a category menu and executed top-to-bottom.

That made **Option A — the full functional clone — the choice**, since "full fidelity" turned out to mean a card stack rather than a node editor. It shipped as `vtea_napari.widgets.ProtocolBuilderWidget` over a headless `vtea_core.workflow` `Step`/`Pipeline` engine, so the same pipeline runs identically from the GUI or from a script/notebook — which also delivers Option B's scriptability rather than trading it away.

The one genuine surprise was the parameter forms. The plan assumed `magicgui` would generate Edit dialogs straight from each step function's type hints. In practice `vtea_core` uses `from __future__ import annotations`, which leaves those hints as plain strings at runtime and breaks magicgui's resolution (it fails trying to import annotation fragments as modules). The port builds forms from `inspect.signature()` with plain `qtpy` widgets instead — still generated from the real signatures, just not via magicgui.

Saved Java workflow XML still needs an import converter so existing user pipelines aren't stranded; that remains open and is tracked under Phase 5.

## The exploration/gating workbench — much smaller than its file count

`vtea.exploration` + `vteaexploration` is 116 files / ~25K LOC, the second-largest area. Reading it before porting found most of that mass is not live domain complexity:

- **One real gate type, not four.** `Gate` is an interface, but `PolygonGate` is the only implementation that works. `RectangleGate` and `FreeFormGate` exist as classes whose every method body is `throw new UnsupportedOperationException`, and neither is instantiated anywhere. A rectangle gate is in practice a 4-vertex polygon built by `GateLayer`'s drag handler.
- **`GateManager.java` and `microGateManager.java` are dead code**, despite their names implying they are the gate-management UI. `GateManager` is constructed once by `ProtocolManagerMulti.addGateManager()` and never shown or populated; `microGateManager` is a ~2800-line near-verbatim fork of ImageJ's `RoiManager` that operates on `ij.gui.Roi` rather than VTEA's gate model and is never instantiated at all. The gate list users actually see is `TableWindow` ("Gate Management").
- **One gate-math operator.** `GateMath`/`AbstractGateMath` is a full SciJava plugin-discovery framework, but `AND` is its only concrete implementer, so the "Classify by Gate Math…" dropdown only ever has one working entry. In the port this is plain `&`/`|`/`~` on boolean arrays — which also supplies the OR/NOT that were never implemented here.
- **No real gate hierarchy.** "Subgating" opened an entirely new `MicroExplorer` window over a pre-filtered dataset rather than nesting gates. The port models parent/child directly (`Gate.parent_id`, a child's membership intersected with its parent's), replacing the window-per-subgate workaround.
- **`vtea.lut` is scatter-plot point coloring, not image display.** It builds a JFreeChart `LookupPaintScale` (11 discrete bands) to color plot points by a third feature — unrelated to ImageJ's per-channel image LUTs, which napari's own layer controls already provide. The port uses a continuous matplotlib colormap picker.
- The ~25 single-method listener interfaces (`AddGateListener`, `PolygonSelectionListener`, `SubGateListener`, `ImageHighlightSelectionListener`, …) exist because Java has no first-class callbacks; they collapse into a handful of Qt signals.

The genuinely valuable, working piece here is the **gallery view** (`GalleryViewWindow`/`GalleryImageProcessor`) — per-object crops around each centroid, click a thumbnail to highlight that object — which ported directly.

This all shipped as the "Object Explorer" dock widget (`ExplorerWidget`, plus `ScatterPlotWidget`/`GateTableWidget`/`GalleryWidget`) over a `vtea_core.gates.Gate`/`GateSet` model.

## Phased roadmap

| Phase | Content | Status |
|---|---|---|
| **0. Foundations & parity harness** | `vtea-core`/`vtea-napari` package skeletons, CI, dependency choices locked in. Golden-dataset regression harness to diff Python outputs against Java reference outputs. | **Done.** Harness lives at `tests/golden/` in the port repo; the Java side of it is `GoldenFixtureGenerator` + `.github/workflows/generate-golden-fixtures.yml` **in this repo**. |
| **1. Core data model & I/O** | `VolumeDataset` (NumPy in-memory + Dask/Zarr chunked) replacing `vtea.dataset`+`vtea.partition`+`vtea.io.zarr`; object model as labeled arrays + DuckDB/pandas measurement tables replacing `vteaobjects.MicroObject`+H2. | **Done.** TIFF + Zarr readers/writers; `bioio` proprietary-format readers not wired up yet. |
| **2. Algorithm core** (largest phase) | Segmentation, measurement extraction (`regionprops_table`-based), clustering, DR, gating, image preprocessing. | **Done.** Segmentation collapsed to composable primitives (threshold → label → optional watershed split → size filter) rather than porting ~15 overlapping Java classes, most of which were working around the lack of one fast native 3D connected-components implementation. |
| **3. Deep learning consolidation** | Replace the JavaCPP VAE/CNN stack and the Py4J Cellpose bridge with native Python. | **Done, but not as a separate module** — see "What the implementation changed". Cellpose landed in `vtea_core.segmentation`, the CNN classifier in a new `vtea_core.classification`. `bioimageio.core` generic model inference is deferred. |
| **4. napari plugin (GUI)** | Protocol builder, `MicroExplorer` equivalent, gate manager, LUTs, thumbnails. | **Done.** Option A protocol builder (`ProtocolBuilderWidget`) and the Object Explorer (`ExplorerWidget` + scatter plot / gate table / gallery), both registered as real npe2 dock widgets and covered by tests that load them through an actual `napari.Viewer`. |
| **— Standalone runtime** (added, not in the original plan) | PyInstaller single-folder bundles so end users don't need a Python install. | **Done.** Linux + Windows, built and smoke-tested in CI, published automatically to GitHub Releases under an auto-incremented `vMAJOR.MINOR.PATCH` tag. |
| **5. Parity validation & cutover** | Run the golden-dataset suite end-to-end: segmentation IoU, feature-table numeric diffs, cluster-assignment ARI, against Java outputs. Beta with real users. Docs + workflow-XML migration converter. | **Current front.** The Python-side comparison utilities and parity tests are written and passing; they self-skip until `tests/golden/fixtures/` is populated. **Populating it is the next concrete step and it starts in this repo** — see below. |
| **6. Decommission Java** | Archive/tag this codebase, update the Fiji update site listing to point at the new pip/conda package and napari plugin index. | Not started — gated on Phase 5. |

Phases 0–4 came in at roughly the low end of the original 23–37 engineer-week estimate, mainly because the two areas feared largest (the protocol builder and the exploration/gating workbench) turned out to carry far less live complexity than their file counts implied — see the two sections above.

## Other risks / open questions

- **Numerical parity for custom algorithms** — partly resolved, partly deliberately changed. `Xmeans` and `GMeansClust` are both thin wrappers around Smile's own X-Means/G-Means, i.e. two different routes to the same goal (choose `k` automatically); the port consolidates them into a single BIC-based `auto_k_kmeans()`. **Exact label-for-label parity with Smile is therefore not expected for those two**, and the parity suite should compare chosen-`k` and cluster-assignment ARI rather than demanding identical output. Deterministic annealing and the kD-tree boundary stitching are **not ported yet** and still need logic-for-logic treatment.
- **Chunked large-volume behavior** — still open. Dask's overlap/stitching semantics differ from the hand-rolled `Chunk`/`ObjectStitcher` system, and `ObjectStitcher`'s cross-chunk object-merge logic has no Python equivalent yet. Validate object counts/boundaries on a volume too large for RAM before trusting it in production.
- **ImageJ macro compatibility** — `vtea.imageprocessing.builtin.IJMacro` lets users embed arbitrary ImageJ1 macros in a pipeline step. Decide explicitly: drop macro support (replace with scikit-image equivalents), or keep an optional `pyimagej` bridge for legacy macro compatibility during the transition window.
- **Format coverage** — confirm `bioio`'s JVM-backed readers cover every vendor format current users rely on before dropping Bio-Formats from the primary code path.
- **User pipeline continuity** — saved `.xml` workflow/protocol files from the Java app should have a conversion path into the new format (called out in Phase 5) so existing collaborators aren't blocked mid-migration.

## What the implementation changed

Building the port corrected several assumptions made in this document. Recorded here so the Java-side analysis above isn't read as still-authoritative:

- **The protocol builder is not drag-and-drop**, so Option A was chosen and was far cheaper than the estimate assumed. See its section above.
- **The exploration/gating workbench is mostly inert.** Dead classes, unimplemented gate types, and a one-operator "plugin framework" account for much of its 25K LOC. See its section above.
- **Deep learning did not become its own module.** The plan implied a `deeplearning` package mirroring `vtea.deeplearning`. That structure only existed in Java because Cellpose (a segmentation method) and the VAE/CNN stack (a classifier) happened to share a binding layer — remove the bindings and they have nothing in common. Cellpose lives in `vtea_core.segmentation` next to the other volume→label-mask functions; the supervised classifier lives in `vtea_core.classification`, parallel to `clustering`/`reduction`. Nothing imports a "deep learning" namespace.
- **`magicgui` didn't survive contact with `from __future__ import annotations`** — parameter forms are built from `inspect.signature()` with plain `qtpy` widgets instead.
- **H2 → DuckDB held up**, but is used less than expected: napari `Labels` layers carry a `.features` DataFrame natively, which is the idiomatic place for a per-object table, so the GUI path doesn't need a SQL layer at all. `MeasurementStore` remains for headless/large-table work.
- **Segmentation collapsed rather than transferred.** The ~15 Java segmentation classes were largely different workarounds for the same problem (no fast native 3D connected components); scikit-image/scipy provide it directly, so they reduce to a handful of composable primitives plus Dask for large volumes.

## Next step — and it starts in this repo

Phase 5 (parity validation) is the current front, and its first concrete action is on the **Java** side:

1. Run the **"Generate golden-dataset fixtures"** workflow in this repository (Actions → Run workflow). It builds this codebase, runs `vtea.tools.GoldenFixtureGenerator` against the two in-repo sample TIFFs plus the deterministic synthetic clustering/PCA inputs, and uploads a `golden-fixtures` artifact. It needs normal internet access to `maven.scijava.org`, which is why it's a CI workflow rather than something run inline.
2. Download that artifact and drop its contents into `tests/golden/fixtures/` in [vtea-python](https://github.com/winfrees/vtea-python). The parity tests there currently self-skip for lack of fixtures; once populated they compare segmentation IoU, feature-table numerics, and cluster-assignment ARI against these Java outputs.
3. Widen coverage: the current generator uses `SingleThreshold3D`, which by construction emits exactly one object, so it exercises the I/O → segmentation → measurement path end-to-end but is not a multi-object fixture. Adding a multi-object segmentation fixture is the highest-value follow-up, since object-splitting behaviour is where Java and Python are most likely to diverge.

Remaining open items unchanged from the original analysis: the workflow-XML import converter, `bioimageio.core` generic model inference, ImageJ macro compatibility (`IJMacro`), ImageJ ROI-file import, `vtea.spatial` statistics, and linear unmixing.

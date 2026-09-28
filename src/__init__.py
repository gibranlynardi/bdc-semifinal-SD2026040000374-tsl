"""Hidden-stratification audit: label-free clustering of frozen DINOv3 embeddings of E-waste photos.

Modules
-------
utils     configuration, paths, logging, library versions
data      load the embeddings, select the de-duplicated Electronic rows, build feature variants
cluster   kNN graph + Leiden, K-Means, UMAP + HDBSCAN
validity  internal indices, resampling and seed stability
select    the label-free choice rule
audit     post-hoc audit against filename names (split scores, orphans, refit stability)
figures   tidy per-photo export (clusters.csv) and the ForceAtlas2 layout; plotting is done elsewhere

Import the modules through the package (``from src import select``). The module ``select`` shares its
name with a standard-library module, so do not put ``src/`` itself on ``sys.path``.
"""

__version__ = "1.0.0"

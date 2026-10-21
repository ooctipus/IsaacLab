* Made the Sharpa demos use task-only studio backdrops in Newton GL by default, with full scenery available through ``--backdrop`` and retained by default in Kit and OVRTX.
* Skipped visual-only Newton shape imports when the Sharpa demos use Kit or OVRTX without Newton GL.
* Kept coffee particle positions in NumPy arrays to avoid repeated scalar-list copying during startup.
* Used flat surface normals for the compact RJ45 patch panel and sockets.

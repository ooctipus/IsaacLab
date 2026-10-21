* Made the coffee and teapot demos use task-only studio backdrops in Newton GL by default, with full scenery available through ``--backdrop`` and retained by default in Kit and OVRTX.
* Limited the RJ45 demo's default Newton GL view to the detailed gold switch front and all RJ45 ports, with one full rack available through ``--backdrop datacenter``, a simpler patch panel through ``--backdrop studio``, and the full data center retained by default in Kit and OVRTX.
* Skipped visual-only Newton shape imports when the Sharpa demos use Kit or OVRTX without Newton GL.
* Kept coffee particle positions in NumPy arrays to avoid repeated scalar-list copying during startup.
* Used flat surface normals for the optional compact RJ45 studio patch panel and sockets.

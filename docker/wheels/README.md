# Pinned integration wheels

`manifest.json` identifies private release assets by immutable asset ID and SHA256.
`buildnpush.sh` fetches missing wheels and verifies existing ones before checking
the lockfile. Generated wheels are ignored by Git and copied into the image at
the same project-relative path used by `uv.lock`.

For private Git dependencies and release assets, pass `--git-netrc /path/to/netrc`
with the credential file outside this checkout and its Docker build context.
The file must contain GitHub credentials for `github.com` and `api.github.com`.
Docker receives it as a BuildKit secret only during dependency installation;
the image and cluster runtime retain no credentials. Never put tokens in source
URLs, build arguments, or this directory.

For a local checkout, call `prepare_dependency_wheels` from `tools/buildnpush.py`
before the first `uv sync`. The experimental Warp wheel targets Linux x86-64
with glibc 2.38 or newer. Other platforms retain the registry dependency and do
not support this experimental capture integration. Use a fresh `WARP_CACHE_PATH`
after changing the pinned wheel: its native ABI differs from upstream Warp.
This wheel is built from the rebased 1.19.0.dev0 integration source.

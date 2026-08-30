# npm architecture

Status: IMPLEMENTED

apps/web is a standalone npm project. Its reproducibility boundary is:

- apps/web/package.json
- apps/web/package-lock.json, lockfile version 3
- Node.js 22.13.0 or newer
- npm ci executed from apps/web

There is no root JavaScript workspace, root package-lock.json, pnpm workspace, or pnpm lockfile.
CI and deployment must run npm commands with apps/web as the working directory. No package
versions were intentionally changed during the readiness repair.

## Five extraneous packages

After a clean npm ci, npm ls reports:

- @emnapi/core
- @emnapi/runtime
- @emnapi/wasi-threads
- @napi-rs/wasm-runtime
- @tybys/wasm-util

All five are optional WASM support packages in package-lock.json. They are bundled dependencies
of the optional @tailwindcss/oxide-wasm32-wasi package and also support optional Rolldown WASM
fallbacks. On Windows x64, npm excludes the wasm32 parent by CPU selector but leaves its bundled
runtime packages at the top level and labels them extraneous. A fresh npm ci reproduces this
state. It does not indicate missing direct dependencies, pnpm contamination, or application
version drift. They are left unchanged.

The clean install also reports eleven dependency advisories. No forced audit update was applied
because that would change versions outside this repair's evidence-backed scope.

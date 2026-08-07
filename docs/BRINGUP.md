# Hardware bring-up

The ordinary workspace test suite is offline. `ipu-tests` builds a trivial
package through `ipu-codegen`, round-trips it through `ipu-package`, loads it,
and checks that every supervisor and worker context halts.

Run it with:

```sh
IPU_CONFIG=config.bin \
POPLAR_SDK_ENABLED=/path/to/poplar \
scripts/hardware-e2e.sh
```

Optional variables:

- `IPU_DEVICE` selects the device node and defaults to `/dev/ipu0`.
- `IPU_TEST_PACKAGE` selects the generated package path and defaults to
  `/tmp/ipu-trivial.ipuexe`.

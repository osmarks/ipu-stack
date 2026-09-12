`profile-schema-v5.ipuexe` was encoded from `profile-schema-v5.txt` using the
application schema at commit f571a62, before profiling records were shared:

```
capnp encode schemas/application.capnp Application < profile-schema-v5.txt > profile-schema-v5.ipuexe
```

It covers every step/activity kind, metadata, epochs, paired and unpaired
transfers, fanout and exchange event cycles. Retain the binary when changing
schemas: regenerating it with the current schema defeats the compatibility test.

At consolidation, encoding this text with the shared schema produced identical
bytes. The aliases preserve field ordinals and data/pointer layouts; Cap'n Proto
stores those layouts, rather than struct type IDs, in ordinary struct pointers
([encoding specification](https://capnproto.org/encoding.html#structs)).

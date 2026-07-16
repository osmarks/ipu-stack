@0xd3bce605d3d25b4f;

enum BlobCodec {
  raw @0;
  zstd @1;
}

struct Blob {
  sha256 @0 :Data;
  uncompressedSize @1 :UInt64;
  codec @2 :BlobCodec;
  data @3 :Data;
}

struct Segment {
  address @0 :UInt32;
  memorySize @1 :UInt32;
  blobIndex @2 :UInt32;
  blobOffset @3 :UInt64;
  fileSize @4 :UInt32;
  flags @5 :UInt32;
}

struct TileImage {
  physicalTile @0 :UInt32;
  entryPoint @1 :UInt32;
  segments @2 :List(Segment);
  commandAddress @3 :UInt32;
  diagnosticAddress @4 :UInt32;
}

struct RegionSlice {
  tile @0 :UInt32;
  tileAddress @1 :UInt32;
  fileOffset @2 :UInt64;
  size @3 :UInt64;
}

struct Binding {
  name @0 :Text;
  dtype @1 :Text;
  shape @2 :List(UInt32);
  slices @3 :List(RegionSlice);
}

struct HostPage {
  index @0 :UInt32;
  size @1 :UInt64;
}

struct HostSlice {
  page @0 :UInt32;
  pageOffset @1 :UInt64;
  fileOffset @2 :UInt64;
  size @3 :UInt64;
}

struct HostCall {
  name @0 :Text;
  command @1 :UInt32;
  phases @2 :UInt32;
  inputs @3 :List(HostSlice);
  outputs @4 :List(HostSlice);
}

struct HostExchange {
  startupMark @0 :UInt32;
  commandPage @1 :UInt32;
  commandOffset @2 :UInt64;
  pages @3 :List(HostPage);
  attachOrder @4 :List(UInt32);
  calls @5 :List(HostCall);
}

struct EntryPoint {
  name @0 :Text;
  command @1 :UInt32;
  externalSyncs @2 :UInt32;
}

struct Application {
  schemaVersion @0 :UInt32;
  compilerVersion @1 :Text;
  target @2 :Text;
  tileMemoryBase @3 :UInt32;
  tileMemorySize @4 :UInt32;
  blobs @5 :List(Blob);
  tiles @6 :List(TileImage);
  inputs @7 :List(Binding);
  outputs @8 :List(Binding);
  weights @9 :List(Binding);
  hostExchange @10 :HostExchange;
  entryPoints @11 :List(EntryPoint);
  buildDigest @12 :Data;
}

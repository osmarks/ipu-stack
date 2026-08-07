@0xd3bce605d3d25b4f;

struct Segment {
  address @0 :UInt32;
  memorySize @1 :UInt32;
  data @2 :Data;
  flags @3 :UInt32;
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
  invocations @5 :UInt32 = 1;
  inputBatchEnds @6 :List(UInt32);
  outputBatchEnds @7 :List(UInt32);
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

struct DeviceConfigWrite {
  offset @0 :UInt32;
  value @1 :UInt32;
}

struct Application {
  schemaVersion @0 :UInt32;
  compilerVersion @1 :Text;
  target @2 :Text;
  tileMemoryBase @3 :UInt32;
  tileMemorySize @4 :UInt32;
  tiles @5 :List(TileImage);
  inputs @6 :List(Binding);
  outputs @7 :List(Binding);
  weights @8 :List(Binding);
  hostExchange @9 :HostExchange;
  entryPoints @10 :List(EntryPoint);
  deviceConfigWrites @11 :List(DeviceConfigWrite);
}

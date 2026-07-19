@0xea2e645066b652a7;

struct MemoryRegion {
  address @0 :UInt32;
  size @1 :UInt32;
  category @2 :Text;
  name @3 :Text;
  hasTensor @4 :Bool;
  tensor @5 :UInt64;
  liveFrom @6 :UInt64;
  liveUntil @7 :UInt64;
}

struct TileMemory {
  logicalTile @0 :UInt32;
  physicalTile @1 :UInt32;
  regions @2 :List(MemoryRegion);
}

struct MemoryProfile {
  schemaVersion @0 :UInt32;
  memoryBase @1 :UInt32;
  memorySize @2 :UInt32;
  tiles @3 :List(TileMemory);
}

@0xbadb9c24f7721fa3;

using Common = import "profile_common.capnp";

struct CycleSample {
  step @0 :Common.Step;
  startCycle @1 :UInt32;
  endCycle @2 :UInt32;
}

struct TileProfile {
  physicalTile @0 :UInt32;
  samples @1 :List(CycleSample);
}

struct Profile {
  schemaVersion @0 :UInt32;
  clockHz @1 :UInt64;
  tiles @2 :List(TileProfile);
}

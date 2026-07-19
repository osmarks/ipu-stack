@0xbadb9c24f7721fa3;

enum StepKind {
  exchange @0;
  compute @1;
}

struct ProfileStep {
  localIndex @0 :UInt32;
  phase @1 :UInt32;
  epoch @2 :UInt32;
  operation @3 :Text;
  kind @4 :StepKind;
  kernel @5 :Text;
  metadata @6 :List(MetadataEntry);
}

struct MetadataEntry {
  name @0 :Text;
  value @1 :Text;
}

struct CycleSample {
  step @0 :ProfileStep;
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

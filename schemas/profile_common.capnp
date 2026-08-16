@0xf3c4b874bcfe20c9;

enum StepKind {
  exchange @0;
  compute @1;
  synchronization @2;
  idle @3;
}

enum ExchangeActivityKind {
  send @0;
  receive @1;
  partnerBusy @2;
}

struct ExchangeActivity {
  kind @0 :ExchangeActivityKind;
  startCycle @1 :UInt32;
  endCycle @2 :UInt32;
}

struct Metadata {
  name @0 :Text;
  value @1 :Text;
}

struct Step {
  localIndex @0 :UInt32;
  phase @1 :UInt32;
  epoch @2 :UInt32;
  operation @3 :Text;
  kind @4 :StepKind;
  kernel @5 :Text;
  metadata @6 :List(Metadata);
  exchangeActivities @7 :List(ExchangeActivity);
  exchangeEventCycles @8 :UInt32;
}

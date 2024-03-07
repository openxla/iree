// RUN: iree-opt --split-input-file --iree-hal-resolve-device-promises %s --mlir-print-local-scope --verify-diagnostics | FileCheck %s

// Resolves device promises.

// CHECK: module @module
// CHECK-SAME: stream.affinity = #hal.device.affinity<@device0, [1, 2, 3]>
module @module attributes {
  stream.affinity = #hal.device.promise<@device0, [1, 2, 3]>
} {
  util.global private @device0 = #hal.device.target<"vmvx"> : !hal.device
  util.global private @device1 = #hal.device.target<"vmvx"> : !hal.device
  // CHECK: util.func private @func
  // CHECK-SAME: stream.affinity = #hal.device.affinity<@device1>
  util.func private @func() -> () attributes {
    stream.affinity = #hal.device.promise<@device1>
  } {
    util.return
  }
}

// -----

// Verifies that promised devices exist.

module @module {
  util.global private @device = #hal.device.target<"vmvx"> : !hal.device
  // expected-error@+1 {{op references a promised device that was not declared}}
  util.func private @func() -> () attributes {
    stream.affinity = #hal.device.promise<@unknown_device>
  } {
    util.return
  }
}

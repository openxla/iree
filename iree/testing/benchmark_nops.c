// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Create no-ops for the functions in benchmark.h

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/testing/benchmark.h"

int64_t iree_benchmark_get_range(iree_benchmark_state_t* state,
                                 iree_host_size_t ordinal) {
  return 0;
}

bool iree_benchmark_keep_running(iree_benchmark_state_t* state,
                                 uint64_t batch_count) {
  return false;
}

void iree_benchmark_skip(iree_benchmark_state_t* state, const char* message) {
  return;
}

void iree_benchmark_pause_timing(iree_benchmark_state_t* state) { return; }

void iree_benchmark_resume_timing(iree_benchmark_state_t* state) { return; }

void iree_benchmark_set_label(iree_benchmark_state_t* state,
                              const char* label) {
  return;
}

void iree_benchmark_set_bytes_processed(iree_benchmark_state_t* state,
                                        int64_t bytes) {
  return;
}

void iree_benchmark_set_items_processed(iree_benchmark_state_t* state,
                                        int64_t items) {
  return;
}

void iree_benchmark_register(iree_string_view_t name,
                             const iree_benchmark_def_t* benchmark_def) {
  return;
}

void iree_benchmark_initialize(int* argc, char** argv) { return; }

void iree_benchmark_run_specified(void) { return; }

// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/loaders/system_library_loader.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/internal/dynamic_library.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/local_executable.h"
#include "iree/hal/local/local_executable_layout.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/dylib_executable_def_reader.h"
#include "iree/schemas/dylib_executable_def_verifier.h"

//===----------------------------------------------------------------------===//
// Verification and file utilities
//===----------------------------------------------------------------------===//

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime.
static iree_status_t iree_hal_dylib_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  // Special handling for valid but mismatching flatbuffers.
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16 ||
      !flatbuffers_has_identifier(flatbuffer_data.data,
                                  iree_DyLibExecutableDef_file_identifier)) {
    return iree_status_from_code(IREE_STATUS_CANCELLED);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_DyLibExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_DyLibExecutableDef_table_t executable_def =
      iree_DyLibExecutableDef_as_root(flatbuffer_data.data);

  if (!flatbuffers_uint8_vec_len(
          iree_DyLibExecutableDef_library_embedded_get(executable_def))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable library_embedded is missing/empty");
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_system_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_system_executable_t {
  iree_hal_local_executable_t base;

  // Flatbuffer definition referencing the executable memory.
  iree_DyLibExecutableDef_table_t def;

  // Loaded platform dynamic library.
  iree_dynamic_library_t* handle;

  // Name used for the file field in tracy and debuggers.
  iree_string_view_t identifier;

  // Queried metadata from the library.
  union {
    const iree_hal_executable_library_header_t** header;
    const iree_hal_executable_library_v0_t* v0;
  } library;

  iree_hal_local_executable_layout_t* layouts[];
} iree_hal_system_executable_t;

extern const iree_hal_local_executable_vtable_t
    iree_hal_system_executable_vtable;

static iree_status_t iree_hal_system_executable_extract_and_load(
    iree_hal_system_executable_t* executable, iree_allocator_t host_allocator) {
  flatbuffers_uint8_vec_t embedded_library_vec =
      iree_DyLibExecutableDef_library_embedded_get(executable->def);
  IREE_RETURN_IF_ERROR(iree_dynamic_library_load_from_memory(
      iree_make_cstring_view("aot"),
      iree_make_const_byte_span(
          embedded_library_vec,
          flatbuffers_uint8_vec_len(embedded_library_vec)),
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &executable->handle));

  flatbuffers_string_t debug_database_filename =
      iree_DyLibExecutableDef_debug_database_filename_get(executable->def);
  flatbuffers_uint8_vec_t debug_database_embedded_vec =
      iree_DyLibExecutableDef_debug_database_embedded_get(executable->def);
  if (flatbuffers_string_len(debug_database_filename) &&
      flatbuffers_uint8_vec_len(debug_database_embedded_vec)) {
    IREE_RETURN_IF_ERROR(iree_dynamic_library_attach_symbols_from_memory(
        executable->handle,
        iree_make_const_byte_span(
            debug_database_embedded_vec,
            flatbuffers_uint8_vec_len(debug_database_embedded_vec))));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_system_executable_query_library(
    iree_hal_system_executable_t* executable) {
  // Get the exported symbol used to get the library metadata.
  iree_hal_executable_library_query_fn_t query_fn = NULL;
  IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(
      executable->handle, IREE_HAL_EXECUTABLE_LIBRARY_EXPORT_NAME,
      (void**)&query_fn));

  // Query for a compatible version of the library.
  executable->library.header =
      query_fn(IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION, /*reserved=*/NULL);
  if (!executable->library.header) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable does not support this version of the runtime (%d)",
        IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION);
  }
  const iree_hal_executable_library_header_t* header =
      *executable->library.header;

  // Ensure that if the library is built for a particular sanitizer that we also
  // were compiled with that sanitizer enabled.
  switch (header->sanitizer) {
    case IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE:
      // Always safe even if the host has a sanitizer enabled; it just means
      // that we won't be able to catch anything from within the executable,
      // however checks outside will (often) still trigger when guard pages are
      // dirtied/etc.
      break;
#if defined(IREE_SANITIZER_ADDRESS)
    case IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_ADDRESS:
      // ASAN is compiled into the host and we can load this library.
      break;
#else
    case IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_ADDRESS:
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "executable library is compiled with ASAN support but the host "
          "runtime is not compiled with it enabled; add -fsanitize=address to "
          "the runtime compilation options");
#endif  // IREE_SANITIZER_ADDRESS
    default:
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "executable library requires a sanitizer the host runtime is not "
          "compiled to enable/understand: %u",
          (uint32_t)header->sanitizer);
  }

  executable->identifier = iree_make_cstring_view(header->name);

  executable->base.dispatch_attrs = executable->library.v0->exports.attrs;

  return iree_ok_status();
}

static int iree_hal_system_executable_import_thunk_v0(
    iree_hal_executable_import_v0_t fn_ptr, void* import_params) {
  return fn_ptr(import_params);
}

// Resolves all of the imports declared by the executable using the given
// |import_provider|.
static iree_status_t iree_hal_system_executable_resolve_imports(
    iree_hal_system_executable_t* executable,
    const iree_hal_executable_import_provider_t import_provider) {
  const iree_hal_executable_import_table_v0_t* import_table =
      &executable->library.v0->imports;
  if (!import_table->count) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Pass all imports right through.
  executable->base.import_thunk = iree_hal_system_executable_import_thunk_v0;

  // Allocate storage for the imports.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(
              executable->base.host_allocator,
              import_table->count * sizeof(*executable->base.imports),
              (void**)&executable->base.imports));

  // Try to resolve each import.
  // NOTE: imports are sorted alphabetically and if we cared we could use this
  // information to more efficiently resolve the symbols from providers (O(n)
  // walk vs potential O(nlogn)/O(n^2)).
  for (uint32_t i = 0; i < import_table->count; ++i) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_executable_import_provider_resolve(
            import_provider, iree_make_cstring_view(import_table->symbols[i]),
            (void**)&executable->base.imports[i]));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_system_executable_create(
    iree_DyLibExecutableDef_table_t executable_def,
    iree_host_size_t executable_layout_count,
    iree_hal_executable_layout_t* const* executable_layouts,
    const iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_def);
  IREE_ASSERT_ARGUMENT(!executable_layout_count || executable_layouts);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_system_executable_t* executable = NULL;
  iree_host_size_t total_size =
      sizeof(*executable) +
      executable_layout_count * sizeof(*executable->layouts);
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  if (iree_status_is_ok(status)) {
    iree_hal_local_executable_initialize(
        &iree_hal_system_executable_vtable, executable_layout_count,
        executable_layouts, &executable->layouts[0], host_allocator,
        &executable->base);
    executable->def = executable_def;
  }
  if (iree_status_is_ok(status)) {
    // Attempt to extract the embedded flatbuffer library and load it.
    // Will scribble information into executable.
    // This is bad, but ehh all this is getting deleted soon and hopefully we
    // can avoid ever touching the disk at all.
    status =
        iree_hal_system_executable_extract_and_load(executable, host_allocator);
  }
  if (iree_status_is_ok(status)) {
    // Query metadata and get the entry point function pointers.
    status = iree_hal_system_executable_query_library(executable);
  }
  if (iree_status_is_ok(status)) {
    // Resolve imports, if any.
    status =
        iree_hal_system_executable_resolve_imports(executable, import_provider);
  }
  if (iree_status_is_ok(status)) {
    // Check to make sure that the entry point count matches the layouts
    // provided.
    if (executable->library.v0->exports.count != executable_layout_count) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "executable provides %u entry points but caller "
          "provided %zu; must match",
          executable->library.v0->exports.count, executable_layout_count);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_release((iree_hal_executable_t*)executable);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_system_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_system_executable_t* executable =
      (iree_hal_system_executable_t*)base_executable;
  iree_allocator_t host_allocator = executable->base.host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_dynamic_library_release(executable->handle);

  iree_hal_local_executable_deinitialize(
      (iree_hal_local_executable_t*)base_executable);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_system_executable_issue_call(
    iree_hal_local_executable_t* base_executable, iree_host_size_t ordinal,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_vec3_t* workgroup_id, iree_byte_span_t local_memory) {
  iree_hal_system_executable_t* executable =
      (iree_hal_system_executable_t*)base_executable;
  const iree_hal_executable_library_v0_t* library = executable->library.v0;

  if (IREE_UNLIKELY(ordinal >= library->exports.count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry point ordinal out of bounds");
  }

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  iree_string_view_t entry_point_name = iree_string_view_empty();
  if (library->exports.names != NULL) {
    entry_point_name = iree_make_cstring_view(library->exports.names[ordinal]);
  }
  if (iree_string_view_is_empty(entry_point_name)) {
    entry_point_name = iree_make_cstring_view("unknown_dylib_call");
  }
  IREE_TRACE_ZONE_BEGIN_EXTERNAL(
      z0, executable->identifier.data, executable->identifier.size, ordinal,
      entry_point_name.data, entry_point_name.size, NULL, 0);
  if (library->exports.tags != NULL) {
    const char* tag = library->exports.tags[ordinal];
    if (tag) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, tag);
    }
  }
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  int ret = library->exports.ptrs[ordinal](dispatch_state, workgroup_id,
                                           local_memory.data);

  IREE_TRACE_ZONE_END(z0);

  return ret == 0 ? iree_ok_status()
                  : iree_make_status(
                        IREE_STATUS_INTERNAL,
                        "executable entry point returned catastrophic error %d",
                        ret);
}

const iree_hal_local_executable_vtable_t iree_hal_system_executable_vtable = {
    .base =
        {
            .destroy = iree_hal_system_executable_destroy,
        },
    .issue_call = iree_hal_system_executable_issue_call,
};

//===----------------------------------------------------------------------===//
// iree_hal_system_library_loader_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_system_library_loader_t {
  iree_hal_executable_loader_t base;
  iree_allocator_t host_allocator;
} iree_hal_system_library_loader_t;

extern const iree_hal_executable_loader_vtable_t
    iree_hal_system_library_loader_vtable;

iree_status_t iree_hal_system_library_loader_create(
    iree_hal_executable_import_provider_t import_provider,
    iree_allocator_t host_allocator,
    iree_hal_executable_loader_t** out_executable_loader) {
  IREE_ASSERT_ARGUMENT(out_executable_loader);
  *out_executable_loader = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_system_library_loader_t* executable_loader = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable_loader), (void**)&executable_loader);
  if (iree_status_is_ok(status)) {
    iree_hal_executable_loader_initialize(
        &iree_hal_system_library_loader_vtable, import_provider,
        &executable_loader->base);
    executable_loader->host_allocator = host_allocator;
    *out_executable_loader = (iree_hal_executable_loader_t*)executable_loader;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_system_library_loader_destroy(
    iree_hal_executable_loader_t* base_executable_loader) {
  iree_hal_system_library_loader_t* executable_loader =
      (iree_hal_system_library_loader_t*)base_executable_loader;
  iree_allocator_t host_allocator = executable_loader->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_loader);

  IREE_TRACE_ZONE_END(z0);
}

#if defined(IREE_PLATFORM_APPLE)
#define IREE_PLATFORM_DYLIB_TYPE "dylib"
#elif defined(IREE_PLATFORM_WINDOWS)
#define IREE_PLATFORM_DYLIB_TYPE "dll"
#else
#define IREE_PLATFORM_DYLIB_TYPE "elf"
#endif  // IREE_PLATFORM_*

static bool iree_hal_system_library_loader_query_support(
    iree_hal_executable_loader_t* base_executable_loader,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_equal(
      executable_format,
      iree_make_cstring_view("system-" IREE_PLATFORM_DYLIB_TYPE "-" IREE_ARCH));
}

static iree_status_t iree_hal_system_library_loader_try_load(
    iree_hal_executable_loader_t* base_executable_loader,
    const iree_hal_executable_spec_t* executable_spec,
    iree_hal_executable_t** out_executable) {
  iree_hal_system_library_loader_t* executable_loader =
      (iree_hal_system_library_loader_t*)base_executable_loader;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify and fetch the executable flatbuffer wrapper.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_dylib_executable_flatbuffer_verify(
                                        executable_spec->executable_data));
  iree_DyLibExecutableDef_table_t executable_def =
      iree_DyLibExecutableDef_as_root(executable_spec->executable_data.data);

  // Perform the load (and requisite disgusting hackery).
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_system_executable_create(
              executable_def, executable_spec->executable_layout_count,
              executable_spec->executable_layouts,
              base_executable_loader->import_provider,
              executable_loader->host_allocator, out_executable));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

const iree_hal_executable_loader_vtable_t
    iree_hal_system_library_loader_vtable = {
        .destroy = iree_hal_system_library_loader_destroy,
        .query_support = iree_hal_system_library_loader_query_support,
        .try_load = iree_hal_system_library_loader_try_load,
};

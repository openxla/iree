// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOTE: the best kind of synchronization is no synchronization; always try to
// design your algorithm so that you don't need anything from this file :)
// See https://travisdowns.github.io/blog/2020/07/06/concurrency-costs.html

#ifndef IREE_BASE_INTERNAL_SYNCHRONIZATION_H_
#define IREE_BASE_INTERNAL_SYNCHRONIZATION_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

// NOTE: clang cannot support thread annotations in C code due to some
// representational bugs... which means that we can't use it here. Boo.
// There's some workarounds I've seen but getting TSAN working would be much
// easier as a starting point.
#if 0  // defined(IREE_COMPILER_CLANG)
#define IREE_THREAD_ANNOTATION_ATTRIBUTE(x) __attribute__((x))
#else
#define IREE_THREAD_ANNOTATION_ATTRIBUTE(x)
#endif  // IREE_COMPILER_CLANG

#ifdef __cplusplus
// Documents if a shared field or global variable needs to be protected by a
// mutex. IREE_GUARDED_BY() allows the user to specify a particular mutex that
// should be held when accessing the annotated variable.
#define IREE_GUARDED_BY(x) IREE_THREAD_ANNOTATION_ATTRIBUTE(guarded_by(x))
#else
#define IREE_GUARDED_BY(x)
#endif  // __cplusplus

#ifdef __cplusplus
// Like IREE_GUARDED_BY but specifies that the contents of a pointer are guarded
// by a mutex instead of the pointer itself.
#define IREE_PTR_GUARDED_BY(x) \
  IREE_THREAD_ANNOTATION_ATTRIBUTE(pt_guarded_by(x))
#else
#define IREE_PTR_GUARDED_BY(x)
#endif  // __cplusplus

// Allow users to fully disable all synchronization for systems that are known
// to never need it. This removes our dependency on pthreads.
#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

// NOTE: we only support futex when not using tsan as we need to add annotations
// for tsan to understand what we are doing.
// https://github.com/llvm-mirror/compiler-rt/blob/master/include/sanitizer/tsan_interface.h
#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_EMSCRIPTEN) || \
    defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_WINDOWS)
#if !defined(IREE_SANITIZER_THREAD)
#define IREE_PLATFORM_HAS_FUTEX 1
#endif  // !IREE_SANITIZER_THREAD
#endif  // IREE_PLATFORM_*

#if defined(IREE_PLATFORM_APPLE)
#include <os/lock.h>
#endif  // IREE_PLATFORM_APPLE

#if !defined(IREE_PLATFORM_WINDOWS)
#include <pthread.h>
#endif  // !IREE_PLATFORM_WINDOWS

// We have the CRITICAL_SECTION path for now but Slim Reader/Writer lock (SRW)
// is much better (and what std::mutex uses). SRW doesn't spin, though, and has
// some other implications that don't quite line up with pthread_mutex_t on most
// platforms. Once we have larger end-to-end benchmarks we should choose based
// on workloads.
#define IREE_MUTEX_USE_WIN32_SRW 1

#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

#ifdef __cplusplus
extern "C" {
#endif

#define IREE_ALL_WAITERS INT32_MAX
#define IREE_INFINITE_TIMEOUT_MS UINT32_MAX

//==============================================================================
// iree_mutex_t
//==============================================================================

// A normal fat mutex (ala std::mutex).
// This may be implemented as a slim mutex on certain platforms but in the worst
// case will be the native platform primitive (like pthread_mutex_t) and as such
// should not be embedded in structures meant to be kept small.
//
// Windows: Slim Reader/Writer (SRW) Locks
// All others: pthread_mutex_t
typedef struct iree_mutex_t IREE_THREAD_ANNOTATION_ATTRIBUTE(
    capability("mutex")) {
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  int reserved;
#elif defined(IREE_PLATFORM_WINDOWS) && defined(IREE_MUTEX_USE_WIN32_SRW)
  SRWLOCK value;
#elif defined(IREE_PLATFORM_WINDOWS)
  CRITICAL_SECTION value;
#else
  pthread_mutex_t value;
#endif  // IREE_PLATFORM_*
#if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_SLOW_LOCKS)
  uint32_t lock_id;
#endif  // IREE_TRACING_FEATURE_SLOW_LOCKS
} iree_mutex_t;

#if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_SLOW_LOCKS)
// Initializes |out_mutex| to the well-defined unlocked contents.
// Must be called prior to using any other iree_mutex_* method.
#define iree_mutex_initialize(out_mutex)                                      \
  static const iree_tracing_location_t TracyConcat(                           \
      __tracy_source_location, __LINE__) = {NULL, __FUNCTION__, __FILE__,     \
                                            (uint32_t)__LINE__, 0};           \
  iree_mutex_initialize_impl(&TracyConcat(__tracy_source_location, __LINE__), \
                             out_mutex);
void iree_mutex_initialize_impl(const iree_tracing_location_t* src_loc,
                                iree_mutex_t* out_mutex);
#else
// Initializes |out_mutex| to the well-defined unlocked contents.
// Must be called prior to using any other iree_mutex_* method.
void iree_mutex_initialize(iree_mutex_t* out_mutex);
#endif  // IREE_TRACING_FEATURE_SLOW_LOCKS

// Deinitializes |mutex| (after a prior call to iree_mutex_initialize).
// The mutex must not be held by any thread.
void iree_mutex_deinitialize(iree_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(locks_excluded(mutex));

// Locks the |mutex| and returns when held by the caller.
void iree_mutex_lock(iree_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(acquire_capability(mutex));

// Tries to lock the |mutex| and returns true if the caller holds the lock.
bool iree_mutex_try_lock(iree_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(try_acquire_capability(true, mutex));

// Unlocks the |mutex|, which must be held by the caller.
void iree_mutex_unlock(iree_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(release_capability(mutex));

//==============================================================================
// iree_slim_mutex_t
//==============================================================================

// TODO(benvanik): instrument with tracy; need to capture source location on
// init and add storage for ID.

// A lightweight unfair lock.
// Depending on platform this is significantly smaller than a mutex (4-8 bytes
// vs 64+ bytes), can always be statically initialized/requires no allocations,
// and performs the minimal amount of work possible while still playing nicely
// with the OS thread scheduler.
//
// Unlike a full mutex these don't have the ability to be shared across
// processes (not something we care about), don't have a way to define timeouts,
// and have only a binary held/unheld state. They are often an order of
// magnitude faster in uncontended/lightly-contended code and the same
// performance in highly-contended code, though, so it's worth it for locks that
// be guarding small data structures (queue pointers, etc) and touched from many
// threads. Since they are so lightweight it's possible to embed them per-object
// instead of per-manager and change from a single highly-contended lock to
// thousands of almost completely uncontended slim locks.
//
// Though these locks support spinning they always have a fallback path that
// ends up calling into the kernel to properly wait the thread. This is critical
// to avoid pathological cases under contention and allowing for thread priority
// inheritance when there are multiple threads competing that may otherwise be
// scheduled in a potentially livelocking order.
//
// The "unfair" here comes from the fact that it's possible on certain platforms
// for certain threads to never be able to acquire the lock in cases of
// extremely high contention or widely disparate thread priority levels. This is
// mitigated by ensuring only very small regions of code are guarded and that
// there's enough work happening outside of the lock on any particular thread to
// ensure that there's some chance of other threads being able to acquire it.
//
// MacOS/iOS: os_unfair_lock
//   Spins and after a short backoff drops to a futex-like behavior of waiting
//   in the kernel. Unfortunately real futexes aren't supported.
// See:
//   https://developer.apple.com/documentation/os/synchronization
//   https://opensource.apple.com/source/libplatform/libplatform-125/src/os/lock.c.auto.html
//
// Emscripten: emscripten_futex_wait/emscripten_futex_wake
//   Spins and after a short backoff drops to a futex-like behavior of waiting
//   in the kernel.
// See:
//   https://github.com/emscripten-core/emscripten/blob/b43474f55aeb49083b9df74fdd0e52ec8decf788/system/include/emscripten/threading.h#L114-L120
//   https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Atomics/wait
//   https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Atomics/notify
//
// Windows: WaitOnAddress/WakeByAddress*
//   Spins and after a short backoff drops to a futex and waits in the kernel.
// See:
//   https://docs.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitonaddress
//   https://devblogs.microsoft.com/oldnewthing/20170601-00/?p=96265
//
// Linux/Android/others: futex
//   Spins and after a short backoff drops to a futex and waits in the kernel.
// See:
//   http://locklessinc.com/articles/futex_cheat_sheet/
//   https://man7.org/linux/man-pages/man2/futex.2.html
//   https://eli.thegreenplace.net/2018/basics-of-futexes/
//   https://bartoszmilewski.com/2008/09/01/thin-lock-vs-futex/
typedef struct iree_slim_mutex_t IREE_THREAD_ANNOTATION_ATTRIBUTE(
    capability("mutex")) {
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  int reserved;
#elif (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_FAST_LOCKS)
  iree_mutex_t impl;  // re-route to slow mutex
#elif defined(IREE_PLATFORM_APPLE)
  os_unfair_lock value;
#elif defined(IREE_PLATFORM_WINDOWS) && defined(IREE_MUTEX_USE_WIN32_SRW)
  SRWLOCK value;
#elif defined(IREE_PLATFORM_HAS_FUTEX)
  iree_atomic_int32_t value;
#else
  iree_mutex_t impl;  // fallback
#endif  // IREE_PLATFORM_*
} iree_slim_mutex_t;

#if (IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_FAST_LOCKS)
// Initializes |out_mutex| to the well-defined unlocked contents.
// Must be called prior to using any other iree_slim_mutex_* method.
#define iree_slim_mutex_initialize(out_mutex)                             \
  static const iree_tracing_location_t TracyConcat(                       \
      __tracy_source_location, __LINE__) = {NULL, __FUNCTION__, __FILE__, \
                                            (uint32_t)__LINE__, 0};       \
  iree_slim_mutex_initialize_impl(                                        \
      &TracyConcat(__tracy_source_location, __LINE__), out_mutex);
void iree_slim_mutex_initialize_impl(const iree_tracing_location_t* src_loc,
                                     iree_slim_mutex_t* out_mutex);
#else
// Initializes |out_mutex| to the well-defined unlocked contents.
// Must be called prior to using any other iree_slim_mutex_* method.
//
// Though optional (static initialization is fine) this is required to support
// lock tracing. Assume it's (mostly) free and always call it if possible. This
// also allows us to swap in a non-slim lock for enhanced debugging if we run
// into threading issues.
void iree_slim_mutex_initialize(iree_slim_mutex_t* out_mutex);
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_FAST_LOCKS

// Deinitializes |mutex| (after a prior call to iree_slim_mutex_initialize).
// The mutex must not be held by any thread.
void iree_slim_mutex_deinitialize(iree_slim_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(locks_excluded(mutex));

// Locks the |mutex| and returns when held by the caller.
void iree_slim_mutex_lock(iree_slim_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(acquire_capability(mutex));

// Tries to lock the |mutex| and returns true if the caller holds the lock.
bool iree_slim_mutex_try_lock(iree_slim_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(try_acquire_capability(true, mutex));

// Unlocks the |mutex|, which must be held by the caller.
void iree_slim_mutex_unlock(iree_slim_mutex_t* mutex)
    IREE_THREAD_ANNOTATION_ATTRIBUTE(release_capability(mutex));

//==============================================================================
// iree_notification_t
//==============================================================================

// TODO(benvanik): add tracy support for watching the waits.

// A lightweight wait-free cross-thread notification mechanism.
// Classically called an 'event counter', these replace the use of condvars in
// lock-free code where you wouldn't want to guard a lock-free data structure
// with a lock.
//
// See:
// http://www.1024cores.net/home/lock-free-algorithms/eventcounts
// https://software.intel.com/en-us/forums/intel-threading-building-blocks/topic/299245
// https://github.com/r10a/Event-Counts
// https://github.com/facebook/folly/blob/master/folly/experimental/EventCount.h
// https://github.com/concurrencykit/ck/blob/master/include/ck_ec.h
typedef struct iree_notification_t {
#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
  // Nothing required. Unused field to make compilers happy.
  int reserved;
#elif !defined(IREE_PLATFORM_HAS_FUTEX)
  // No futex on darwin/when using TSAN, so use mutex/condvar instead.
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  uint32_t epoch;
  uint32_t waiters;
#else
  iree_atomic_int64_t value;
#endif  // IREE_PLATFORM_*
} iree_notification_t;

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
#define IREE_NOTIFICATION_INIT \
  { IREE_ATOMIC_VAR_INIT(0) }
#elif !defined(IREE_PLATFORM_HAS_FUTEX)
#define IREE_NOTIFICATION_INIT \
  { PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0 }
#else
#define IREE_NOTIFICATION_INIT \
  { IREE_ATOMIC_VAR_INIT(0) }
#endif  // notification type

// Initializes a notification to no waiters and an initial epoch of 0.
void iree_notification_initialize(iree_notification_t* out_notification);

// Deinitializes |notification| (after a prior call to
// iree_notification_initialize). No threads may be waiting on the notification.
void iree_notification_deinitialize(iree_notification_t* notification);

// Notifies up to |count| waiters of a change. Each waiter will wake and can
// check to see if they need to do any additional work.
// To notify all potential waiters pass IREE_ALL_WAITERS.
//
// Acts as (at least) a memory_order_release store on `notification`.
//
// https://en.cppreference.com/w/cpp/atomic/memory_order#Release-Acquire_ordering
//
// Example:
//
//   Thread 1:
//
//     *some_value = 123;  // ordinary memory write (no atomic required)
//     iree_notification_post(notification, ...);  // "release" atomic op
//
//   Thread 2:
//
//     iree_notification_commit_wait(notification, ...);  // "acquire" atomic op
//     return *some_value;  // ordinary memory read (no atomic required)
//
// Here it is guaranteed that Thread 2 sees the value 123 written by Thread 1.
//
// On the Thread 1 side, the `*some_value = 123` cannot be reordered past the
// iree_notification_post, thanks to `release` ordering.
// On the Thread 2 side, the `return some_value` cannot be reordered before the
// iree_notification_commit_wait thanks to `acquire` ordering.
// Between Thread 1 and Thread 2, the ordering is provided by the release and
// acquire operations on the same `notification` object.
void iree_notification_post(iree_notification_t* notification, int32_t count);

typedef uint32_t iree_wait_token_t;  // opaque

// Prepares for a wait operation, returning a token that must be passed to
// iree_notification_commit_wait to perform the actual wait.
//
// Acts as a memory_order_acq_rel (i.e. both acquire and release) load-store on
// `notification`. See iree_notification_post comment.
iree_wait_token_t iree_notification_prepare_wait(
    iree_notification_t* notification);

// Commits a pending wait operation when the caller has ensured it must wait.
// Waiting will continue until a notification has been posted or |deadline_ns|
// is reached. Returns false if the deadline is reached before a notification is
// posted.
//
// Acts as (at least) a memory_order_acquire load on `notification`.
// See iree_notification_post comment.
bool iree_notification_commit_wait(iree_notification_t* notification,
                                   iree_wait_token_t wait_token,
                                   iree_time_t deadline_ns);

// Cancels a pending wait operation without blocking.
//
// Acts as (at least) a memory_order_relaxed operation on `notification`:
// There are no synchronization or ordering constraints
// imposed on other reads or writes, only this operation's atomicity is
// guaranteed.
void iree_notification_cancel_wait(iree_notification_t* notification);

// Returns true if the condition is true.
// |arg| is the |condition_arg| passed to the await function.
// Implementations must ensure they are coherent with their state values.
typedef bool (*iree_condition_fn_t)(void* arg);

// Blocks and waits until |condition_fn| returns true. Other threads must modify
// state checked by the |condition_fn| and post the notification.
// Returns true if the condition is true before |timeout| is reached. If the
// timeout is infinite then the return will always be true.
//
// Example:
//  thread 1:
//   bool check_flag_pred(void* arg) {
//     return iree_atomic_int32_load((iree_atomic_int32_t*)arg,
//                                   iree_memory_order_acquire) == 1;
//   }
//   iree_atomic_int32_t* flag = ...;
//   iree_notification_await(&notification, check_flag_pred, flag);
//  thread 2:
//   iree_atomic_int32_store(flag, 1, iree_memory_order_release);
//   iree_notification_post(&notification, IREE_ALL_WAITERS);
bool iree_notification_await(iree_notification_t* notification,
                             iree_condition_fn_t condition_fn,
                             void* condition_arg, iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_SYNCHRONIZATION_H_

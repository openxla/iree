// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/PluginManager.h"

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

// Declare entrypoints for each statically registered plugin.
#define HANDLE_PLUGIN_ID(plugin_id)                          \
  extern "C" bool iree_register_compiler_plugin_##plugin_id( \
      mlir::iree_compiler::PluginRegistrar *);
#include "iree/compiler/PluginAPI/Config/StaticLinkedPlugins.inc"
#undef HANDLE_PLUGIN_ID

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::EmptyPluginOptions);
IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::PluginManagerOptions);

namespace mlir::iree_compiler {

AbstractPluginRegistration::~AbstractPluginRegistration() = default;
AbstractPluginSession::~AbstractPluginSession() = default;

void PluginManagerOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category("IREE compiler plugin options");

  binder.list<std::string>("iree-plugin", plugins,
                           llvm::cl::desc("Plugins to activate"),
                           llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-print-plugin-info", printPluginInfo,
      llvm::cl::desc("Prints available and activated plugin info to stderr"),
      llvm::cl::cat(category));
}

PluginManager::PluginManager() {}

bool PluginManager::loadAvailablePlugins() {
// Initialize static plugins.
#define HANDLE_PLUGIN_ID(plugin_id) \
  if (!iree_register_compiler_plugin_##plugin_id(this)) return false;
#include "iree/compiler/PluginAPI/Config/StaticLinkedPlugins.inc"
#undef HANDLE_PLUGIN_ID
  return true;
}

void PluginManager::globalInitialize() {
  for (auto &kv : registrations) {
    kv.second->globalInitialize();
  }
}

void PluginManager::initializeCLI() {
  for (auto &kv : registrations) {
    kv.second->initializeCLI();
  }
}

void PluginManager::registerDialects(DialectRegistry &registry) {
  for (auto &kv : registrations) {
    kv.second->registerDialects(registry);
  }
}

void PluginRegistrar::registerPlugin(
    std::unique_ptr<AbstractPluginRegistration> registration) {
  std::string_view id = registration->getPluginId();
  auto foundIt = registrations.insert(
      std::make_pair(llvm::StringRef(id), std::move(registration)));
  if (!foundIt.second) {
    llvm::errs() << "ERROR: Duplicate plugin registration for '" << id << "'\n";
    abort();
  }
}

LogicalResult PluginManagerSession::activatePlugins() {
  auto getAvailableIds = [&]() -> llvm::SmallVector<llvm::StringRef> {
    llvm::SmallVector<llvm::StringRef> availableIds;
    for (auto &kv : pluginManager.registrations) {
      availableIds.push_back(kv.first());
    }
    std::sort(availableIds.begin(), availableIds.end());
    return availableIds;
  };

  // Print available plugins.
  if (options.printPluginInfo) {
    // Get the available plugins.
    llvm::errs() << "[IREE plugins]: Available plugins: ";
    llvm::interleaveComma(getAvailableIds(), llvm::errs());
    llvm::errs() << "\n";
  }

  // Process activations.
  // In the future, we may make this smarter by allowing dependencies and
  // sorting accordingly. For now, what you say is what you get.
  llvm::StringSet<> activatedPluginIds;
  for (auto &pluginId : options.plugins) {
    if (options.printPluginInfo) {
      llvm::errs() << "[IREE plugins]: Activating plugin '" << pluginId
                   << "'\n";
    }
    if (!activatedPluginIds.insert(pluginId).second) {
      if (options.printPluginInfo) {
        llvm::errs() << "[IREE plugins]: Skipping duplicate plugin '"
                     << pluginId << "'\n";
      }
      continue;
    }
    auto foundIt = pluginManager.registrations.find(pluginId);
    if (foundIt == pluginManager.registrations.end()) {
      auto diag = mlir::emitError(mlir::UnknownLoc::get(context))
                  << "Could not activate requested IREE plugin '" << pluginId
                  << "' because it is not registered. Available plugins: ";
      llvm::interleaveComma(getAvailableIds(), diag);
      return failure();
    }

    std::unique_ptr<AbstractPluginSession> instance =
        foundIt->second->createSession(context);
    if (failed(instance->activate())) return failure();
    activatedSessions.push_back(std::move(instance));
  }

  return success();
}

}  // namespace mlir::iree_compiler

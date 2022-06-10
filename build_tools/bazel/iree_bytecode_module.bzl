# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for compiling IREE executables, modules, and archives."""

load("//build_tools/embed_data:build_defs.bzl", "c_embed_data")

# TODO(benvanik): port to a full starlark rule, document, etc.

def iree_bytecode_module(
        name,
        src,
        module_name = None,
        flags,
        compile_tool = "//tools:iree-compile",
        linker_tool = "@llvm-project//lld:lld",
        c_identifier = "",
        deps = [],
        **kwargs):
    """Builds an IREE bytecode module.

    Args:
        name: Name of the target
        src: mlir source file to be compiled to an IREE module.
        flags: additional flags to pass to the compiler.
            `--output-format=vm-bytecode` is included automatically.
        compile_tool: the compiler to use to generate the module.
            Defaults to iree-compile.
        linker_tool: the linker to use.
            Defaults to the lld from the llvm-project directory.
        module_name: Optional name for the generated IREE module.
            Defaults to `name.vmfb`.
        c_identifier: Optional. Enables embedding the module as C data.
        deps: Optional. Dependencies to add to the generated library.
        **kwargs: any additional attributes to pass to the underlying rules.
    """

    if not module_name:
        module_name = "%s.vmfb" % (name)

    native.genrule(
        name = name,
        srcs = [src],
        outs = [
            module_name,
        ],
        cmd = " && ".join([
            " ".join([
                "$(location %s)" % (compile_tool),
                " ".join(["--output-format=vm-bytecode"] + flags),
                "--iree-llvm-embedded-linker-path=$(location %s)" % (linker_tool),
                "--iree-llvm-wasm-linker-path=$(location %s)" % (linker_tool),
                # Note: --iree-llvm-system-linker-path is left unspecified.
                "-o $(location %s)" % (module_name),
                "$(location %s)" % (src),
            ]),
        ]),
        tools = [compile_tool, linker_tool],
        message = "Compiling IREE module %s..." % (name),
        output_to_bindir = 1,
        **kwargs
    )

    # Embed the module for use in C.
    if c_identifier:
        c_embed_data(
            name = "%s_c" % (name),
            identifier = c_identifier,
            srcs = [module_name],
            c_file_output = "%s_c.c" % (name),
            h_file_output = "%s_c.h" % (name),
            flatten = True,
            deps = deps,
            **kwargs
        )

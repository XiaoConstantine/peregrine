const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Compile every Metal shader into peregrine.metallib and install it.
    const install_metallib = addMetallib(b);
    b.getInstallStep().dependOn(&install_metallib.step);

    // The installed metallib path, baked into the binary so the runtime can
    // load its default library without a CLI flag.
    const options = b.addOptions();
    const metallib_path = b.allocator.dupeZ(u8, b.getInstallPath(.lib, "peregrine.metallib")) catch @panic("oom");
    options.addOption([:0]const u8, "metallib_path", metallib_path);
    // Library module (the package surface). It owns the Objective-C Metal
    // bridge and links the Apple frameworks; consumers get them transitively.
    const mod = b.addModule("peregrine", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    addMetalRuntime(b, mod, options);

    // The CLI executable.
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "peregrine", .module = mod },
        },
    });
    const exe = b.addExecutable(.{
        .name = "peregrine",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    // `zig build run`
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run the peregrine CLI");
    run_step.dependOn(&run_cmd.step);

    // `zig build fmt-check`
    const fmt = b.addFmt(.{ .paths = &.{ "src", "build.zig" }, .check = true });
    const fmt_step = b.step("fmt-check", "Check Zig formatting");
    fmt_step.dependOn(&fmt.step);

    // `zig build check` (compile only, no install)
    const check_step = b.step("check", "Compile-check without installing");
    check_step.dependOn(&exe.step);

    // `zig build test`
    const unit_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const cli_tests = b.addTest(.{
        .root_module = exe_mod,
    });
    const run_cli_tests = b.addRunArtifact(cli_tests);
    const test_step = b.step("test", "Run Zig unit tests");
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_cli_tests.step);
    const runtime_context_test_files = [_][]const u8{
        "src/model_unit_tests.zig",
    };
    for (runtime_context_test_files) |file| {
        const runtime_test_mod = b.createModule(.{
            .root_source_file = b.path(file),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        addMetalRuntime(b, runtime_test_mod, options);
        const runtime_tests = b.addTest(.{
            .root_module = runtime_test_mod,
        });
        const run_runtime_tests = b.addRunArtifact(runtime_tests);
        test_step.dependOn(&run_runtime_tests.step);
    }

    // `zig build script-check`
    const shell_check = b.addSystemCommand(&.{ "bash", "-n" });
    shell_check.addFileArg(b.path("tools/run-qwen35"));
    const python_check = b.addSystemCommand(&.{ "python3", "-m", "py_compile" });
    const python_scripts = [_][]const u8{
        "tools/gdn_chunk_scan_probe.py",
    };
    for (python_scripts) |script| {
        python_check.addFileArg(b.path(script));
    }
    const script_check_step = b.step("script-check", "Syntax-check shell and Python helper scripts");
    script_check_step.dependOn(&shell_check.step);
    script_check_step.dependOn(&python_check.step);

    // `zig build ci` = fmt-check + tests + compile + script checks + metallib
    const ci_step = b.step("ci", "fmt-check, test, compile, script-check, and build metallib");
    ci_step.dependOn(fmt_step);
    ci_step.dependOn(test_step);
    ci_step.dependOn(check_step);
    ci_step.dependOn(script_check_step);
    ci_step.dependOn(&install_metallib.step);
}

fn addMetalRuntime(b: *std.Build, mod: *std.Build.Module, options: *std.Build.Step.Options) void {
    mod.addIncludePath(b.path("src/runtime"));
    mod.addCSourceFile(.{ .file = b.path("src/runtime/bridge.m"), .flags = &.{} });
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Metal", .{});
    mod.addOptions("build_options", options);
}

/// Compile every `shaders/*.metal` to AIR and link them into
/// `peregrine.metallib`, returning the install step for the linked library.
fn addMetallib(b: *std.Build) *std.Build.Step.InstallFile {
    const shaders = [_][]const u8{
        "fill",
        "linear",
        "linear_decode",
        "rmsnorm",
        "embedding",
        "rope",
        "attention",
        "attention_decode",
        "gated_delta",
        "conv1d",
        "gated_norm",
        "elementwise",
        "layout",
        "argmax",
        "mlx_gemm",
    };

    const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib" });
    for (shaders) |name| {
        // No -ffast-math: correctness first. IEEE compliance keeps kernel
        // outputs faithful enough to match the MLX baseline token-for-token.
        const compile = b.addSystemCommand(&.{
            "xcrun", "-sdk", "macosx", "metal", "-c", "-O3", "-I", "shaders",
        });
        compile.addFileArg(b.path(b.fmt("shaders/{s}.metal", .{name})));
        const air = compile.addPrefixedOutputFileArg("-o", b.fmt("{s}.air", .{name}));
        link.addFileArg(air);
    }
    const metallib = link.addPrefixedOutputFileArg("-o", "peregrine.metallib");
    return b.addInstallLibFile(metallib, "peregrine.metallib");
}

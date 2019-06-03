load ("//:cuda_configure.bzl", "cuda_configure")

# ===== absl =====
http_archive(
    name = "com_google_absl",
    url = "https://github.com/abseil/abseil-cpp/archive/9613678332c976568272c8f4a78631a29159271d.zip",
    sha256 = "bd377bfffee86e65a1aafcb1c4cc1535916199006683d53262f335ae21e67c57",
    strip_prefix = "abseil-cpp-9613678332c976568272c8f4a78631a29159271d",
)

# ===== gtest =====
new_http_archive(
    name = "com_google_googletest",
    url = "https://github.com/google/googletest/archive/88f0493098c8d9fd0f096c2158a0e56deb952d53.zip",
    sha256 = "9b0f8553f8da88ff0e77f0b456f12cec3cb52ddf3797508a00503c16e77745b8",
    strip_prefix = "googletest-88f0493098c8d9fd0f096c2158a0e56deb952d53",
    build_file_content = """
licenses(["notice"])

cc_library(
    name = "gtest",
    srcs = [
        "googletest/src/gtest-all.cc",
    ],
    hdrs = glob([
        "**/*.h",
        "googletest/src/*.cc",
    ]),
    includes = [
        "googletest",
        "googletest/include",
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "gtest_main",
    srcs = ["googletest/src/gtest_main.cc"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
    deps = [":gtest"],
)
    """,
)

# ===== benchmark =====
new_http_archive(
    name = "com_google_benchmark",
    url = "https://github.com/google/benchmark/archive/v1.3.0.zip",
    sha256 = "51c2d2d35491aea83aa6121afc4a1fd9262fbd5ad679eb5e03c9fa481e42571e",
    strip_prefix = "benchmark-1.3.0",
    build_file_content = """
licenses(["notice"])

cc_library(
    name = "benchmark",
    srcs = glob(["src/*.cc"]),
    hdrs = glob([
        "include/**/*.h",
        "src/*.h",
    ]),
    copts = [
        "-DHAVE_POSIX_REGEX",
    ],
    includes = [
        ".",
        "include",
    ],
    linkstatic = 1,
    visibility = [
        "//visibility:public",
    ],
)
    """,
)

# ===== gflags, required by glog =====
http_archive(
    name = "com_github_gflags_gflags",
    url = "https://github.com/gflags/gflags/archive/v2.2.1.zip",
    sha256 = "4e44b69e709c826734dbbbd5208f61888a2faf63f239d73d8ba0011b2dccc97a",
    strip_prefix = "gflags-2.2.1",
)

# ===== glog =====
new_http_archive(
    name = "com_google_glog",
    url = "https://github.com/google/glog/archive/v0.3.5.zip",
    sha256 = "267103f8a1e9578978aa1dc256001e6529ef593e5aea38193d31c2872ee025e8",
    strip_prefix = "glog-0.3.5",
    build_file_content = """
licenses(["notice"])

cc_library(
    name = "glog",
    srcs = [
        "config.h",
        "src/base/commandlineflags.h",
        "src/base/googleinit.h",
        "src/base/mutex.h",
        "src/demangle.cc",
        "src/demangle.h",
        "src/logging.cc",
        "src/raw_logging.cc",
        "src/signalhandler.cc",
        "src/symbolize.cc",
        "src/symbolize.h",
        "src/utilities.cc",
        "src/utilities.h",
        "src/vlog_is_on.cc",
    ] + glob(["src/stacktrace*.h"]),
    hdrs = [
        "src/glog/log_severity.h",
        "src/glog/logging.h",
        "src/glog/raw_logging.h",
        "src/glog/stl_logging.h",
        "src/glog/vlog_is_on.h",
    ],
    copts = [
        "-Wno-sign-compare",
        "-U_XOPEN_SOURCE",
    ],
    includes = ["./src"],
    linkopts = ["-lpthread"] + select({
        ":libunwind": ["-lunwind"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_gflags_gflags//:gflags",
    ],
)

config_setting(
    name = "libunwind",
    values = {
        "define": "libunwind=true",
    },
)

genrule(
    name = "run_configure",
    srcs = [
        "README",
        "Makefile.in",
        "config.guess",
        "config.sub",
        "install-sh",
        "ltmain.sh",
        "missing",
        "libglog.pc.in",
        "src/config.h.in",
        "src/glog/logging.h.in",
        "src/glog/raw_logging.h.in",
        "src/glog/stl_logging.h.in",
        "src/glog/vlog_is_on.h.in",
    ],
    outs = [
        "config.h",
        "src/glog/logging.h",
        "src/glog/raw_logging.h",
        "src/glog/stl_logging.h",
        "src/glog/vlog_is_on.h",
    ],
    cmd = "$(location :configure)" +
          "&& cp -v src/config.h $(location config.h) " +
          "&& cp -v src/glog/logging.h $(location src/glog/logging.h) " +
          "&& cp -v src/glog/raw_logging.h $(location src/glog/raw_logging.h) " +
          "&& cp -v src/glog/stl_logging.h $(location src/glog/stl_logging.h) " +
          "&& cp -v src/glog/vlog_is_on.h $(location src/glog/vlog_is_on.h) ",
    tools = [
        "configure",
    ],
)
    """,
)

# ===== protobuf =====
# proto_library rules implicitly depend on @com_google_protobuf//:protoc
http_archive(
    name = "com_google_protobuf",
    url = "https://github.com/google/protobuf/archive/v3.5.1.zip",
    sha256 = "1f8b9b202e9a4e467ff0b0f25facb1642727cdf5e69092038f15b37c75b99e45",
    strip_prefix = "protobuf-3.5.1",
)

# ===== cuda =====
cuda_configure(name = "local_config_cuda")

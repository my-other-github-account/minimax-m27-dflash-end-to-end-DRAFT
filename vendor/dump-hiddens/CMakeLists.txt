# Vendored dump-hiddens tools.
#
# When this CMakeLists is included from inside the llama.cpp tree (via the
# build script staging it under examples/dump-hiddens/), `${CMAKE_SOURCE_DIR}`
# is the llama.cpp root and ${CMAKE_SOURCE_DIR}/common/ exists and contains
# common.h plus the implementation library.
#
# In recent upstream llama.cpp (master HEAD as of 2026-05) the library target
# was renamed from `common` to `llama-common`. We detect which name exists at
# configure time and link the right one.

if(TARGET llama-common)
    set(_DH_COMMON_LIB llama-common)
elseif(TARGET common)
    set(_DH_COMMON_LIB common)
else()
    message(FATAL_ERROR "dump-hiddens: cannot find common library target (looked for 'llama-common' and 'common')")
endif()

# Common-library headers live at <llama-cpp-root>/common/. Make them
# discoverable from the vendored source files (which #include "common.h").
set(_DH_COMMON_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/common")

set(TARGET llama-dump-hiddens)
add_executable(${TARGET} dump_hiddens.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_include_directories(${TARGET} PRIVATE ${_DH_COMMON_INCLUDE_DIR})
target_link_libraries(${TARGET} PRIVATE ${_DH_COMMON_LIB} llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET2 llama-dump-hiddens-batch)
add_executable(${TARGET2} dump_hiddens_batch.cpp)
install(TARGETS ${TARGET2} RUNTIME)
target_include_directories(${TARGET2} PRIVATE ${_DH_COMMON_INCLUDE_DIR})
target_link_libraries(${TARGET2} PRIVATE ${_DH_COMMON_LIB} llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET2} PRIVATE cxx_std_17)

# Persistent batched-decode worker used by the dflash-llama trace-server
# (src/dflash_llama/tracegen/). Talks JSONL over stdin/stdout. Supports
# multi-prompt batched decoding with n_seq_max=8 — the speedup path
# documented in repro/08-tracegen-server.md (~2.5× vs spawn-per-prompt).
set(TARGET3 llama-dump-hiddens-worker)
add_executable(${TARGET3} dump_hiddens_worker.cpp)
install(TARGETS ${TARGET3} RUNTIME)
target_include_directories(${TARGET3} PRIVATE ${_DH_COMMON_INCLUDE_DIR})
target_link_libraries(${TARGET3} PRIVATE ${_DH_COMMON_LIB} llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET3} PRIVATE cxx_std_17)

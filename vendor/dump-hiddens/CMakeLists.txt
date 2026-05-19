set(TARGET llama-dump-hiddens)
add_executable(${TARGET} dump_hiddens.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET2 llama-dump-hiddens-batch)
add_executable(${TARGET2} dump_hiddens_batch.cpp)
install(TARGETS ${TARGET2} RUNTIME)
target_link_libraries(${TARGET2} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET2} PRIVATE cxx_std_17)

# Persistent batched-decode worker used by the dflash-llama trace-server
# (src/dflash_llama/tracegen/). Talks JSONL over stdin/stdout. Supports
# multi-prompt batched decoding with n_seq_max=8 — the speedup path
# documented in repro/08-tracegen-server.md (~2.5× vs spawn-per-prompt).
set(TARGET3 llama-dump-hiddens-worker)
add_executable(${TARGET3} dump_hiddens_worker.cpp)
install(TARGETS ${TARGET3} RUNTIME)
target_link_libraries(${TARGET3} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET3} PRIVATE cxx_std_17)

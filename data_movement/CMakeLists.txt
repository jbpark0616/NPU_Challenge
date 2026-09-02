set(UNIT_TESTS_DATA_MOVEMENT_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/dm_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dram_unary/test_unary_dram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_to_all/test_one_to_all.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_to_all/test_multicast_schemes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_to_all/test_multicast_exclude.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_to_all/segmented_rest/test_segmented_rest_planner.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_to_all/segmented_rest/test_conv_axis_segmented_mcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_to_all/segmented_rest/test_single_sender_segment_count.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/loopback/test_loopback.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reshard_hardcoded/test_reshard_hardcoded.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deinterleave_hardcoded/test_deinterleave_hardcoded.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv_hardcoded/test_conv_hardcoded.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_to_one/test_one_to_one.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_from_one/test_one_from_one.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/core_bidirectional/test_core_bidirectional.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_from_all/test_one_from_all.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_to_all/test_all_to_all.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/all_from_all/test_all_from_all.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interleaved/test_interleaved.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_packet/test_one_packet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/interleaved_to_sharded_hardcoded/test_interleaved_to_sharded_hardcoded.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_interleaved/test_multi_interleaved.cpp
)

add_executable(unit_tests_data_movement ${UNIT_TESTS_DATA_MOVEMENT_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_data_movement)

target_link_libraries(unit_tests_data_movement PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_data_movement
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_data_movement
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Actual TTNN Conv integration scaffold for selective core replacement. Keep it
# separate from the data-movement microbench executable so TTNN dependencies do
# not affect the existing tests.
add_executable(
    unit_tests_conv_core_replacement
    ${CMAKE_CURRENT_SOURCE_DIR}/conv_core_replacement/test_core_placement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv_core_replacement/test_conv_core_replacement.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/conv_core_replacement/test_segmented_buffer_distribution.cpp
)
target_link_libraries(
    unit_tests_conv_core_replacement
    PRIVATE
        test_metal_common_libs
        TTNN::CPP
)
target_include_directories(
    unit_tests_conv_core_replacement
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/ttnn/unit_tests/gtests
)
set_target_properties(
    unit_tests_conv_core_replacement
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/ttnn
)

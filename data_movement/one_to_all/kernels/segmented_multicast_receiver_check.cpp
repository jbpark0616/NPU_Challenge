// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

// Compile args: destination L1 address, payload bytes, expected word, status
// L1 address, test id. Runtime args: semaphore id.
//
// This receiver checks data immediately after the semaphore wait.  Thus a
// PASS status proves the sender did not expose the semaphore before the data
// multicast was visible in this receiver's L1.
void kernel_main() {
    constexpr uint32_t destination_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t payload_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t expected_word = get_compile_time_arg_val(2);
    constexpr uint32_t status_l1_addr = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);
    const uint32_t semaphore_id = get_arg_val<uint32_t>(0);

    auto* semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(semaphore_id));
    auto* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(destination_l1_addr);
    auto* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_l1_addr);
    {
        DeviceZoneScopedN("Mcast receiver semaphore wait");
        noc_semaphore_wait(semaphore, 1);
    }
    bool payload_matches = true;
    for (uint32_t word = 0; word < payload_bytes / sizeof(uint32_t); ++word) {
        payload_matches &= (payload[word] == expected_word);
    }
    *status = payload_matches ? 1u : 0u;

    DeviceTimestampedData("Number of transactions", 1);
    DeviceTimestampedData("Transaction size in bytes", payload_bytes);
    DeviceTimestampedData("Test id", test_id);
}

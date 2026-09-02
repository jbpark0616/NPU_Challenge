// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

// Runtime args: semaphore id, then one (start_x, start_y, end_x, end_y,
// destination_count) tuple for every compile-time segment.
void kernel_main() {
    constexpr uint32_t source_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t destination_l1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t payload_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_segments = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);
    constexpr uint32_t total_destination_count = get_compile_time_arg_val(5);
    const uint32_t semaphore_id = get_arg_val<uint32_t>(0);

    {
        DeviceZoneScopedN("RISCV0");
        {
            DeviceZoneScopedN("Mcast data issue");
            for (uint32_t segment = 0; segment < num_segments; ++segment) {
                const uint32_t arg = 1 + segment * 5;
                const uint32_t start_x = get_arg_val<uint32_t>(arg + 0);
                const uint32_t start_y = get_arg_val<uint32_t>(arg + 1);
                const uint32_t end_x = get_arg_val<uint32_t>(arg + 2);
                const uint32_t end_y = get_arg_val<uint32_t>(arg + 3);
                const uint32_t destination_count = get_arg_val<uint32_t>(arg + 4);
                const uint64_t destination = noc_index == 0
                    ? get_noc_multicast_addr(start_x, start_y, end_x, end_y, destination_l1_addr)
                    : get_noc_multicast_addr(end_x, end_y, start_x, start_y, destination_l1_addr);
                noc_async_write_multicast(source_l1_addr, destination, payload_bytes, destination_count, false);
            }
        }
        {
            DeviceZoneScopedN("Mcast data barrier");
            noc_async_write_barrier();  // exactly once, after all data segments
        }

        const uint32_t semaphore_l1_addr = get_semaphore(semaphore_id);
        auto* semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_l1_addr);
        *semaphore = 1;  // sender local signal; non-loopback mcast does not write self
        {
            DeviceZoneScopedN("Mcast semaphore issue");
            for (uint32_t segment = 0; segment < num_segments; ++segment) {
                const uint32_t arg = 1 + segment * 5;
                const uint32_t start_x = get_arg_val<uint32_t>(arg + 0);
                const uint32_t start_y = get_arg_val<uint32_t>(arg + 1);
                const uint32_t end_x = get_arg_val<uint32_t>(arg + 2);
                const uint32_t end_y = get_arg_val<uint32_t>(arg + 3);
                const uint32_t destination_count = get_arg_val<uint32_t>(arg + 4);
                const uint64_t destination = noc_index == 0
                    ? get_noc_multicast_addr(start_x, start_y, end_x, end_y, semaphore_l1_addr)
                    : get_noc_multicast_addr(end_x, end_y, start_x, start_y, semaphore_l1_addr);
                noc_semaphore_set_multicast(semaphore_l1_addr, destination, destination_count, false);
            }
        }
    }

    DeviceTimestampedData("Number of transactions", 1);
    DeviceTimestampedData("Transaction size in bytes", payload_bytes);
    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("NoC Index", noc_index);
    DeviceTimestampedData("Segment Count", num_segments);
    DeviceTimestampedData("Destination Count", total_destination_count);
}

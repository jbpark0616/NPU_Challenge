// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t payload_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t test_id = get_compile_time_arg_val(1);
    const uint32_t semaphore_id = get_arg_val<uint32_t>(0);
    auto* semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(semaphore_id));
    {
        DeviceZoneScopedN("RISCV1");
        noc_semaphore_wait(semaphore, 1);
    }
    DeviceTimestampedData("Number of transactions", 1);
    DeviceTimestampedData("Transaction size in bytes", payload_bytes);
    DeviceTimestampedData("Test id", test_id);
}

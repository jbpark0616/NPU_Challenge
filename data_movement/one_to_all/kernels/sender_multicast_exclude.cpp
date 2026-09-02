// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

namespace {

constexpr uint32_t STATUS_MAGIC = 0x4558434Cu;  // "EXCL"
static_assert(4 * NOC_ADDR_NODE_ID_BITS == 24, "Unexpected Blackhole multicast coordinate width");

}  // namespace

// Blackhole-only discovery kernel for NOC_BRCST_EXCLUDE.
//
// This deliberately uses a posted multicast and bounded command-buffer polls.
// It does not wait for multicast acknowledgements, so an incorrect exclude
// encoding cannot deadlock on an expected-ack mismatch. The host must still
// run this test behind an external watchdog on a freshly reset device.
void kernel_main() {
    constexpr uint32_t source_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t destination_l1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t payload_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t rectangle_start_x = get_compile_time_arg_val(3);
    constexpr uint32_t rectangle_start_y = get_compile_time_arg_val(4);
    constexpr uint32_t rectangle_end_x = get_compile_time_arg_val(5);
    constexpr uint32_t rectangle_end_y = get_compile_time_arg_val(6);
    constexpr uint32_t exclude_start_x = get_compile_time_arg_val(7);
    constexpr uint32_t exclude_start_y = get_compile_time_arg_val(8);
    constexpr uint32_t exclude_dir_x = get_compile_time_arg_val(9);
    constexpr uint32_t exclude_dir_y = get_compile_time_arg_val(10);
    constexpr uint32_t actual_num_destinations = get_compile_time_arg_val(11);
    constexpr uint32_t status_l1_addr = get_compile_time_arg_val(12);
    constexpr uint32_t ready_poll_limit = get_compile_time_arg_val(13);
    constexpr uint32_t delivery_delay_cycles = get_compile_time_arg_val(14);

    auto* status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_l1_addr);
    status[0] = STATUS_MAGIC;
    status[1] = 0;  // Initial command-buffer ready.
    status[2] = 0;  // Command accepted after send.
    status[3] = 0;  // Initial-ready poll count.
    status[4] = 0;  // Completion-ready poll count.
    status[5] = 0;  // Exclude encoding used by this NoC instance.
    status[6] = 0xFFFFFFFFu;  // Exclude register after cleanup.

    uint32_t initial_polls = 0;
    while (!noc_cmd_buf_ready(noc_index, write_cmd_buf) && initial_polls < ready_poll_limit) {
        ++initial_polls;
    }
    status[3] = initial_polls;
    if (!noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
        return;
    }
    status[1] = 1;

    const uint64_t destination = noc_index == 0
        ? get_noc_multicast_addr(
              rectangle_start_x,
              rectangle_start_y,
              rectangle_end_x,
              rectangle_end_y,
              destination_l1_addr)
        : get_noc_multicast_addr(
              rectangle_end_x,
              rectangle_end_y,
              rectangle_start_x,
              rectangle_start_y,
              destination_l1_addr);

    // Blackhole exclusion describes a quadrant by an enabled bit, its start
    // coordinate, and one direction bit per axis. It is not a second
    // multicast rectangle.
    const uint32_t exclude_encoding =
        (EXCLUDE_ENABLED << EXCLUDE_ENABLED_OFFSET) |
        (DYNAMIC_NOC_DIRECTION(noc_index, exclude_dir_y) << EXCLUDE_DIRECTION_Y_OFFSET) |
        (DYNAMIC_NOC_DIRECTION(noc_index, exclude_dir_x) << EXCLUDE_DIRECTION_X_OFFSET) |
        (DYNAMIC_NOC_Y(noc_index, exclude_start_y) << EXCLUDE_START_Y_OFFSET) |
        (DYNAMIC_NOC_X(noc_index, exclude_start_x) << EXCLUDE_START_X_OFFSET);
    status[5] = exclude_encoding;

    NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_BRCST_EXCLUDE, exclude_encoding);

    // update_counter=false: do not modify the standard barrier accounting.
    // posted=true: no per-destination acknowledgements are requested.
    ncrisc_noc_fast_write<noc_mode, false, false>(
        noc_index,
        write_cmd_buf,
        source_l1_addr,
        destination,
        payload_bytes,
        NOC_MULTICAST_WRITE_VC,
        true,   // multicast
        false,  // linked
        actual_num_destinations,
        true,   // multicast path reservation
        true);  // posted

    uint32_t completion_polls = 0;
    while (!noc_cmd_buf_ready(noc_index, write_cmd_buf) && completion_polls < ready_poll_limit) {
        ++completion_polls;
    }
    status[4] = completion_polls;
    if (noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
        status[2] = 1;
        // The command buffer is ready again, so the previous command has
        // consumed its state. Clear the sticky register before any later use.
        NOC_CMD_BUF_WRITE_REG(noc_index, write_cmd_buf, NOC_BRCST_EXCLUDE, 0);
        status[6] = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_BRCST_EXCLUDE);
    }

    // Posted-write completion has no acknowledgement. Give a single 64-byte
    // packet ample time to reach L1 before the host reads the destination.
    for (uint32_t cycle = 0; cycle < delivery_delay_cycles; ++cycle) {
        asm volatile("nop");
    }
}

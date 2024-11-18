static int megasas_build_sense(MegasasCmd *cmd, uint8_t *sense_ptr,

    uint8_t sense_len)

{

    uint32_t pa_hi = 0, pa_lo;

    target_phys_addr_t pa;



    if (sense_len > cmd->frame->header.sense_len) {

        sense_len = cmd->frame->header.sense_len;

    }

    if (sense_len) {

        pa_lo = le32_to_cpu(cmd->frame->pass.sense_addr_lo);

        if (megasas_frame_is_sense64(cmd)) {

            pa_hi = le32_to_cpu(cmd->frame->pass.sense_addr_hi);

        }

        pa = ((uint64_t) pa_hi << 32) | pa_lo;

        cpu_physical_memory_write(pa, sense_ptr, sense_len);

        cmd->frame->header.sense_len = sense_len;

    }

    return sense_len;

}

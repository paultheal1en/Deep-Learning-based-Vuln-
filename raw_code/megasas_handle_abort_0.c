static int megasas_handle_abort(MegasasState *s, MegasasCmd *cmd)

{

    uint64_t abort_ctx = le64_to_cpu(cmd->frame->abort.abort_context);

    target_phys_addr_t abort_addr, addr_hi, addr_lo;

    MegasasCmd *abort_cmd;



    addr_hi = le32_to_cpu(cmd->frame->abort.abort_mfi_addr_hi);

    addr_lo = le32_to_cpu(cmd->frame->abort.abort_mfi_addr_lo);

    abort_addr = ((uint64_t)addr_hi << 32) | addr_lo;



    abort_cmd = megasas_lookup_frame(s, abort_addr);

    if (!abort_cmd) {

        trace_megasas_abort_no_cmd(cmd->index, abort_ctx);

        s->event_count++;

        return MFI_STAT_OK;

    }

    if (!megasas_use_queue64(s)) {

        abort_ctx &= (uint64_t)0xFFFFFFFF;

    }

    if (abort_cmd->context != abort_ctx) {

        trace_megasas_abort_invalid_context(cmd->index, abort_cmd->index,

                                            abort_cmd->context);

        s->event_count++;

        return MFI_STAT_ABORT_NOT_POSSIBLE;

    }

    trace_megasas_abort_frame(cmd->index, abort_cmd->index);

    megasas_abort_command(abort_cmd);

    if (!s->event_cmd || abort_cmd != s->event_cmd) {

        s->event_cmd = NULL;

    }

    s->event_count++;

    return MFI_STAT_OK;

}

static void event_scan(PowerPCCPU *cpu, sPAPRMachineState *spapr,

                       uint32_t token, uint32_t nargs,

                       target_ulong args,

                       uint32_t nret, target_ulong rets)

{

    uint32_t mask, buf, len, event_len;

    sPAPREventLogEntry *event;

    struct rtas_error_log *hdr;



    if (nargs != 4 || nret != 1) {

        rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

        return;

    }



    mask = rtas_ld(args, 0);

    buf = rtas_ld(args, 2);

    len = rtas_ld(args, 3);



    event = rtas_event_log_dequeue(mask, false);

    if (!event) {

        goto out_no_events;

    }



    hdr = event->data;

    event_len = be32_to_cpu(hdr->extended_length) + sizeof(*hdr);



    if (event_len < len) {

        len = event_len;

    }



    cpu_physical_memory_write(buf, event->data, len);

    rtas_st(rets, 0, RTAS_OUT_SUCCESS);

    g_free(event->data);

    g_free(event);

    return;



out_no_events:

    rtas_st(rets, 0, RTAS_OUT_NO_ERRORS_FOUND);

}

static void rtas_ibm_os_term(PowerPCCPU *cpu,

                            sPAPRMachineState *spapr,

                            uint32_t token, uint32_t nargs,

                            target_ulong args,

                            uint32_t nret, target_ulong rets)

{

    target_ulong ret = 0;



    qapi_event_send_guest_panicked(GUEST_PANIC_ACTION_PAUSE, &error_abort);



    rtas_st(rets, 0, ret);

}

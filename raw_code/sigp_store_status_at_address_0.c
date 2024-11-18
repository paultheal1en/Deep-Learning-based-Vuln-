static void sigp_store_status_at_address(CPUState *cs, run_on_cpu_data arg)

{

    S390CPU *cpu = S390_CPU(cs);

    SigpInfo *si = arg.host_ptr;

    uint32_t address = si->param & 0x7ffffe00u;



    /* cpu has to be stopped */

    if (s390_cpu_get_state(cpu) != CPU_STATE_STOPPED) {

        set_sigp_status(si, SIGP_STAT_INCORRECT_STATE);

        return;

    }



    cpu_synchronize_state(cs);



    if (s390_store_status(cpu, address, false)) {

        set_sigp_status(si, SIGP_STAT_INVALID_PARAMETER);

        return;

    }

    si->cc = SIGP_CC_ORDER_CODE_ACCEPTED;

}

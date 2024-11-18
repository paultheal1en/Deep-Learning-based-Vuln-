static void sigp_set_prefix(CPUState *cs, run_on_cpu_data arg)

{

    S390CPU *cpu = S390_CPU(cs);

    SigpInfo *si = arg.host_ptr;

    uint32_t addr = si->param & 0x7fffe000u;



    cpu_synchronize_state(cs);



    if (!address_space_access_valid(&address_space_memory, addr,

                                    sizeof(struct LowCore), false)) {

        set_sigp_status(si, SIGP_STAT_INVALID_PARAMETER);

        return;

    }



    /* cpu has to be stopped */

    if (s390_cpu_get_state(cpu) != CPU_STATE_STOPPED) {

        set_sigp_status(si, SIGP_STAT_INCORRECT_STATE);

        return;

    }



    cpu->env.psa = addr;

    cpu_synchronize_post_init(cs);

    si->cc = SIGP_CC_ORDER_CODE_ACCEPTED;

}

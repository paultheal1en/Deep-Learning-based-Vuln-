static void sigp_stop(CPUState *cs, run_on_cpu_data arg)

{

    S390CPU *cpu = S390_CPU(cs);

    SigpInfo *si = arg.host_ptr;



    if (s390_cpu_get_state(cpu) != CPU_STATE_OPERATING) {

        si->cc = SIGP_CC_ORDER_CODE_ACCEPTED;

        return;

    }



    /* disabled wait - sleeping in user space */

    if (cs->halted) {

        s390_cpu_set_state(CPU_STATE_STOPPED, cpu);

    } else {

        /* execute the stop function */

        cpu->env.sigp_order = SIGP_STOP;

        cpu_inject_stop(cpu);

    }

    si->cc = SIGP_CC_ORDER_CODE_ACCEPTED;

}

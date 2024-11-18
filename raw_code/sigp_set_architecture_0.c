static int sigp_set_architecture(S390CPU *cpu, uint32_t param,

                                 uint64_t *status_reg)

{

    CPUState *cur_cs;

    S390CPU *cur_cpu;

    bool all_stopped = true;



    CPU_FOREACH(cur_cs) {

        cur_cpu = S390_CPU(cur_cs);



        if (cur_cpu == cpu) {

            continue;

        }

        if (s390_cpu_get_state(cur_cpu) != CPU_STATE_STOPPED) {

            all_stopped = false;

        }

    }



    *status_reg &= 0xffffffff00000000ULL;



    /* Reject set arch order, with czam we're always in z/Arch mode. */

    *status_reg |= (all_stopped ? SIGP_STAT_INVALID_PARAMETER :

                    SIGP_STAT_INCORRECT_STATE);

    return SIGP_CC_STATUS_STORED;

}

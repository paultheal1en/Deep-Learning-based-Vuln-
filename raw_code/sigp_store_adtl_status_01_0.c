static void sigp_store_adtl_status(void *arg)

{

    SigpInfo *si = arg;



    if (!kvm_check_extension(kvm_state, KVM_CAP_S390_VECTOR_REGISTERS)) {

        set_sigp_status(si, SIGP_STAT_INVALID_ORDER);

        return;

    }



    /* cpu has to be stopped */

    if (s390_cpu_get_state(si->cpu) != CPU_STATE_STOPPED) {

        set_sigp_status(si, SIGP_STAT_INCORRECT_STATE);

        return;

    }



    /* parameter must be aligned to 1024-byte boundary */

    if (si->param & 0x3ff) {

        set_sigp_status(si, SIGP_STAT_INVALID_PARAMETER);

        return;

    }



    cpu_synchronize_state(CPU(si->cpu));



    if (kvm_s390_store_adtl_status(si->cpu, si->param)) {

        set_sigp_status(si, SIGP_STAT_INVALID_PARAMETER);

        return;

    }

    si->cc = SIGP_CC_ORDER_CODE_ACCEPTED;

}

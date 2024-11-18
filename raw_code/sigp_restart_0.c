static void sigp_restart(CPUState *cs, run_on_cpu_data arg)

{

    S390CPU *cpu = S390_CPU(cs);

    SigpInfo *si = arg.host_ptr;

    struct kvm_s390_irq irq = {

        .type = KVM_S390_RESTART,

    };



    switch (s390_cpu_get_state(cpu)) {

    case CPU_STATE_STOPPED:

        /* the restart irq has to be delivered prior to any other pending irq */

        cpu_synchronize_state(cs);

        do_restart_interrupt(&cpu->env);

        s390_cpu_set_state(CPU_STATE_OPERATING, cpu);

        break;

    case CPU_STATE_OPERATING:

        kvm_s390_vcpu_interrupt(cpu, &irq);

        break;

    }

    si->cc = SIGP_CC_ORDER_CODE_ACCEPTED;

}

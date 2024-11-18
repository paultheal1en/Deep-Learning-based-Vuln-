int kvm_arch_handle_exit(CPUState *cs, struct kvm_run *run)

{

    S390CPU *cpu = S390_CPU(cs);

    int ret = 0;



    switch (run->exit_reason) {

        case KVM_EXIT_S390_SIEIC:

            ret = handle_intercept(cpu);

            break;

        case KVM_EXIT_S390_RESET:

            qemu_system_reset_request();

            break;

        case KVM_EXIT_S390_TSCH:

            ret = handle_tsch(cpu);

            break;

        case KVM_EXIT_DEBUG:

            ret = kvm_arch_handle_debug_exit(cpu);

            break;

        default:

            fprintf(stderr, "Unknown KVM exit: %d\n", run->exit_reason);

            break;

    }



    if (ret == 0) {

        ret = EXCP_INTERRUPT;

    }

    return ret;

}

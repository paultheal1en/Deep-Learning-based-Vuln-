int kvm_s390_cpu_restart(S390CPU *cpu)

{

    kvm_s390_interrupt(cpu, KVM_S390_RESTART, 0);

    s390_add_running_cpu(cpu);

    qemu_cpu_kick(CPU(cpu));

    DPRINTF("DONE: KVM cpu restart: %p\n", &cpu->env);

    return 0;

}

static int handle_hypercall(S390CPU *cpu, struct kvm_run *run)

{

    CPUS390XState *env = &cpu->env;



    cpu_synchronize_state(CPU(cpu));

    env->regs[2] = s390_virtio_hypercall(env);



    return 0;

}

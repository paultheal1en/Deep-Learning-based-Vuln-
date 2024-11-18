static void sigp_cpu_start(void *arg)

{

    CPUState *cs = arg;

    S390CPU *cpu = S390_CPU(cs);



    s390_cpu_set_state(CPU_STATE_OPERATING, cpu);

    DPRINTF("DONE: KVM cpu start: %p\n", &cpu->env);

}

void s390x_tod_timer(void *opaque)

{

    S390CPU *cpu = opaque;

    CPUS390XState *env = &cpu->env;



    env->pending_int |= INTERRUPT_TOD;

    cpu_interrupt(CPU(cpu), CPU_INTERRUPT_HARD);

}

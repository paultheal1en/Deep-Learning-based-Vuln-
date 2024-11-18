void do_cpu_init(X86CPU *cpu)

{

    CPUX86State *env = &cpu->env;

    int sipi = env->interrupt_request & CPU_INTERRUPT_SIPI;

    uint64_t pat = env->pat;



    cpu_reset(CPU(cpu));

    env->interrupt_request = sipi;

    env->pat = pat;

    apic_init_reset(env->apic_state);

    env->halted = !cpu_is_bsp(env);

}

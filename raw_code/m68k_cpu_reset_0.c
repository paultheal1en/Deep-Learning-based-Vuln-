static void m68k_cpu_reset(CPUState *s)

{

    M68kCPU *cpu = M68K_CPU(s);

    M68kCPUClass *mcc = M68K_CPU_GET_CLASS(cpu);

    CPUM68KState *env = &cpu->env;



    mcc->parent_reset(s);



    memset(env, 0, offsetof(CPUM68KState, end_reset_fields));

#if !defined(CONFIG_USER_ONLY)

    env->sr = 0x2700;

#endif

    m68k_switch_sp(env);

    /* ??? FP regs should be initialized to NaN.  */

    cpu_m68k_set_ccr(env, 0);

    /* TODO: We should set PC from the interrupt vector.  */

    env->pc = 0;

}

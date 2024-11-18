static void lm32_cpu_reset(CPUState *s)

{

    LM32CPU *cpu = LM32_CPU(s);

    LM32CPUClass *lcc = LM32_CPU_GET_CLASS(cpu);

    CPULM32State *env = &cpu->env;



    if (qemu_loglevel_mask(CPU_LOG_RESET)) {

        qemu_log("CPU Reset (CPU %d)\n", s->cpu_index);

        log_cpu_state(env, 0);

    }



    lcc->parent_reset(s);



    tlb_flush(env, 1);



    /* reset cpu state */

    memset(env, 0, offsetof(CPULM32State, breakpoints));

}

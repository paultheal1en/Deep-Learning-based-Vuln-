void DBDMA_schedule(void)

{

    CPUState *env = cpu_single_env;

    if (env)

        cpu_interrupt(env, CPU_INTERRUPT_EXIT);

}

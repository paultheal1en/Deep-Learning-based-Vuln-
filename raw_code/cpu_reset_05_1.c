void cpu_reset (CPUMIPSState *env)
{
    memset(env, 0, offsetof(CPUMIPSState, breakpoints));
    tlb_flush(env, 1);
    /* Minimal init */
#if defined(CONFIG_USER_ONLY)
    env->hflags = MIPS_HFLAG_UM;
#else
    if (env->hflags & MIPS_HFLAG_BMASK) {
        /* If the exception was raised from a delay slot,
           come back to the jump.  */
        env->CP0_ErrorEPC = env->active_tc.PC - 4;
    } else {
        env->CP0_ErrorEPC = env->active_tc.PC;
    env->active_tc.PC = (int32_t)0xBFC00000;
    env->CP0_Wired = 0;
    /* SMP not implemented */
    env->CP0_EBase = 0x80000000;
    env->CP0_Status = (1 << CP0St_BEV) | (1 << CP0St_ERL);
    /* vectored interrupts not implemented, timer on int 7,
       no performance counters. */
    env->CP0_IntCtl = 0xe0000000;
    {
        int i;
        for (i = 0; i < 7; i++) {
            env->CP0_WatchLo[i] = 0;
            env->CP0_WatchHi[i] = 0x80000000;
        env->CP0_WatchLo[7] = 0;
        env->CP0_WatchHi[7] = 0;
    /* Count register increments in debug mode, EJTAG version 1 */
    env->CP0_Debug = (1 << CP0DB_CNT) | (0x1 << CP0DB_VER);
    env->hflags = MIPS_HFLAG_CP0;
#endif
    env->exception_index = EXCP_NONE;
    cpu_mips_register(env, env->cpu_model);
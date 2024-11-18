void cpu_reset(CPUARMState *env)
{
    uint32_t id;
    id = env->cp15.c0_cpuid;
    memset(env, 0, offsetof(CPUARMState, breakpoints));
    if (id)
        cpu_reset_model_id(env, id);
#if defined (CONFIG_USER_ONLY)
    env->uncached_cpsr = ARM_CPU_MODE_USR;
    env->vfp.xregs[ARM_VFP_FPEXC] = 1 << 30;
#else
    /* SVC mode with interrupts disabled.  */
    env->uncached_cpsr = ARM_CPU_MODE_SVC | CPSR_A | CPSR_F | CPSR_I;
    /* On ARMv7-M the CPSR_I is the value of the PRIMASK register, and is
       clear at reset.  */
    if (IS_M(env))
        env->uncached_cpsr &= ~CPSR_I;
    env->vfp.xregs[ARM_VFP_FPEXC] = 0;
    env->cp15.c2_base_mask = 0xffffc000u;
#endif
    env->regs[15] = 0;
    tlb_flush(env, 1);
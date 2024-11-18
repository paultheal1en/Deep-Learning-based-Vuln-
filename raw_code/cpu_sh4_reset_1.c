static void cpu_sh4_reset(CPUSH4State * env)
{
#if defined(CONFIG_USER_ONLY)
    env->sr = 0;
#else
    env->sr = SR_MD | SR_RB | SR_BL | SR_I3 | SR_I2 | SR_I1 | SR_I0;
#endif
    env->vbr = 0;
    env->pc = 0xA0000000;
#if defined(CONFIG_USER_ONLY)
    env->fpscr = FPSCR_PR; /* value for userspace according to the kernel */
    set_float_rounding_mode(float_round_nearest_even, &env->fp_status); /* ?! */
#else
    env->fpscr = 0x00040001; /* CPU reset value according to SH4 manual */
    set_float_rounding_mode(float_round_to_zero, &env->fp_status);
#endif
    env->mmucr = 0;
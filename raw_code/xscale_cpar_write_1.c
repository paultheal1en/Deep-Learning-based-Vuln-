static void xscale_cpar_write(CPUARMState *env, const ARMCPRegInfo *ri,

                              uint64_t value)

{

    value &= 0x3fff;

    if (env->cp15.c15_cpar != value) {

        /* Changes cp0 to cp13 behavior, so needs a TB flush.  */

        tb_flush(env);

        env->cp15.c15_cpar = value;

    }

}

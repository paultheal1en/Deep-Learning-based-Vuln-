static void csselr_write(CPUARMState *env, const ARMCPRegInfo *ri,

                         uint64_t value)

{

    env->cp15.c0_cssel = value & 0xf;

}

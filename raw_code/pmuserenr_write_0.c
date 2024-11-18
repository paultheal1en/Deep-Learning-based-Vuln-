static void pmuserenr_write(CPUARMState *env, const ARMCPRegInfo *ri,

                            uint64_t value)

{

    env->cp15.c9_pmuserenr = value & 1;

}

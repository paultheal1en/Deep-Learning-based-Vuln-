static void pmsav5_insn_ap_write(CPUARMState *env, const ARMCPRegInfo *ri,

                                 uint64_t value)

{

    env->cp15.c5_insn = extended_mpu_ap_bits(value);

}

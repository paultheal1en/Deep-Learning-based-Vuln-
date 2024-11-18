static void pmsav5_data_ap_write(CPUARMState *env, const ARMCPRegInfo *ri,

                                 uint64_t value)

{

    env->cp15.c5_data = extended_mpu_ap_bits(value);

}

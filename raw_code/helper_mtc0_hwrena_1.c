void helper_mtc0_hwrena(CPUMIPSState *env, target_ulong arg1)

{

    env->CP0_HWREna = arg1 & 0x0000000F;

}

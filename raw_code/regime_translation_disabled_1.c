static inline bool regime_translation_disabled(CPUARMState *env,

                                               ARMMMUIdx mmu_idx)

{

    if (arm_feature(env, ARM_FEATURE_M)) {

        switch (env->v7m.mpu_ctrl[regime_is_secure(env, mmu_idx)] &

                (R_V7M_MPU_CTRL_ENABLE_MASK | R_V7M_MPU_CTRL_HFNMIENA_MASK)) {

        case R_V7M_MPU_CTRL_ENABLE_MASK:

            /* Enabled, but not for HardFault and NMI */

            return mmu_idx == ARMMMUIdx_MNegPri ||

                mmu_idx == ARMMMUIdx_MSNegPri;

        case R_V7M_MPU_CTRL_ENABLE_MASK | R_V7M_MPU_CTRL_HFNMIENA_MASK:

            /* Enabled for all cases */

            return false;

        case 0:

        default:

            /* HFNMIENA set and ENABLE clear is UNPREDICTABLE, but

             * we warned about that in armv7m_nvic.c when the guest set it.

             */

            return true;

        }

    }



    if (mmu_idx == ARMMMUIdx_S2NS) {

        return (env->cp15.hcr_el2 & HCR_VM) == 0;

    }

    return (regime_sctlr(env, mmu_idx) & SCTLR_M) == 0;

}

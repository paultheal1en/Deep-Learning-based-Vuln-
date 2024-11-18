static inline uint32_t regime_el(CPUARMState *env, ARMMMUIdx mmu_idx)

{

    switch (mmu_idx) {

    case ARMMMUIdx_S2NS:

    case ARMMMUIdx_S1E2:

        return 2;

    case ARMMMUIdx_S1E3:

        return 3;

    case ARMMMUIdx_S1SE0:

        return arm_el_is_aa64(env, 3) ? 1 : 3;

    case ARMMMUIdx_S1SE1:

    case ARMMMUIdx_S1NSE0:

    case ARMMMUIdx_S1NSE1:



        return 1;

    default:

        g_assert_not_reached();

    }

}
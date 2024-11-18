static inline bool regime_is_secure(CPUARMState *env, ARMMMUIdx mmu_idx)

{

    switch (mmu_idx) {

    case ARMMMUIdx_S12NSE0:

    case ARMMMUIdx_S12NSE1:

    case ARMMMUIdx_S1NSE0:

    case ARMMMUIdx_S1NSE1:

    case ARMMMUIdx_S1E2:

    case ARMMMUIdx_S2NS:



        return false;

    case ARMMMUIdx_S1E3:

    case ARMMMUIdx_S1SE0:

    case ARMMMUIdx_S1SE1:

        return true;

    default:

        g_assert_not_reached();

    }

}
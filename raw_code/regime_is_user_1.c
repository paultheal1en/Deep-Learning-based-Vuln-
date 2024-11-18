static inline bool regime_is_user(CPUARMState *env, ARMMMUIdx mmu_idx)

{

    switch (mmu_idx) {

    case ARMMMUIdx_S1SE0:

    case ARMMMUIdx_S1NSE0:


        return true;

    default:

        return false;

    case ARMMMUIdx_S12NSE0:

    case ARMMMUIdx_S12NSE1:

        g_assert_not_reached();

    }

}
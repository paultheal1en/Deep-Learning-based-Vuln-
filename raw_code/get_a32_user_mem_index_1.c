static inline int get_a32_user_mem_index(DisasContext *s)

{

    /* Return the core mmu_idx to use for A32/T32 "unprivileged load/store"

     * insns:

     *  if PL2, UNPREDICTABLE (we choose to implement as if PL0)

     *  otherwise, access as if at PL0.

     */

    switch (s->mmu_idx) {

    case ARMMMUIdx_S1E2:        /* this one is UNPREDICTABLE */

    case ARMMMUIdx_S12NSE0:

    case ARMMMUIdx_S12NSE1:

        return arm_to_core_mmu_idx(ARMMMUIdx_S12NSE0);

    case ARMMMUIdx_S1E3:

    case ARMMMUIdx_S1SE0:

    case ARMMMUIdx_S1SE1:

        return arm_to_core_mmu_idx(ARMMMUIdx_S1SE0);




    case ARMMMUIdx_S2NS:

    default:

        g_assert_not_reached();

    }

}
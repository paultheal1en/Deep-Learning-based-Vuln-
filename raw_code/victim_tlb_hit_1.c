static bool victim_tlb_hit(CPUArchState *env, size_t mmu_idx, size_t index,

                           size_t elt_ofs, target_ulong page)

{

    size_t vidx;

    for (vidx = 0; vidx < CPU_VTLB_SIZE; ++vidx) {

        CPUTLBEntry *vtlb = &env->tlb_v_table[mmu_idx][vidx];

        target_ulong cmp = *(target_ulong *)((uintptr_t)vtlb + elt_ofs);



        if (cmp == page) {

            /* Found entry in victim tlb, swap tlb and iotlb.  */

            CPUTLBEntry tmptlb, *tlb = &env->tlb_table[mmu_idx][index];

            CPUIOTLBEntry tmpio, *io = &env->iotlb[mmu_idx][index];

            CPUIOTLBEntry *vio = &env->iotlb_v[mmu_idx][vidx];



            tmptlb = *tlb; *tlb = *vtlb; *vtlb = tmptlb;

            tmpio = *io; *io = *vio; *vio = tmpio;

            return true;

        }

    }

    return false;

}

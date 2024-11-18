static hwaddr ppc_hash64_pteg_search(PowerPCCPU *cpu, hwaddr hash,

                                     uint32_t slb_pshift, bool secondary,

                                     target_ulong ptem, ppc_hash_pte64_t *pte)

{

    CPUPPCState *env = &cpu->env;

    int i;

    uint64_t token;

    target_ulong pte0, pte1;

    target_ulong pte_index;



    pte_index = (hash & env->htab_mask) * HPTES_PER_GROUP;

    token = ppc_hash64_start_access(cpu, pte_index);

    if (!token) {

        return -1;

    }

    for (i = 0; i < HPTES_PER_GROUP; i++) {

        pte0 = ppc_hash64_load_hpte0(cpu, token, i);

        pte1 = ppc_hash64_load_hpte1(cpu, token, i);



        if ((pte0 & HPTE64_V_VALID)

            && (secondary == !!(pte0 & HPTE64_V_SECONDARY))

            && HPTE64_V_COMPARE(pte0, ptem)) {

            uint32_t pshift = ppc_hash64_pte_size_decode(pte1, slb_pshift);

            if (pshift == 0) {

                continue;

            }

            /* We don't do anything with pshift yet as qemu TLB only deals

             * with 4K pages anyway

             */

            pte->pte0 = pte0;

            pte->pte1 = pte1;

            ppc_hash64_stop_access(cpu, token);

            return (pte_index + i) * HASH_PTE_SIZE_64;

        }

    }

    ppc_hash64_stop_access(cpu, token);

    /*

     * We didn't find a valid entry.

     */

    return -1;

}

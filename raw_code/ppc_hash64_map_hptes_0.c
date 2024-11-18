const ppc_hash_pte64_t *ppc_hash64_map_hptes(PowerPCCPU *cpu,

                                             hwaddr ptex, int n)

{

    ppc_hash_pte64_t *hptes = NULL;

    hwaddr pte_offset = ptex * HASH_PTE_SIZE_64;



    if (cpu->env.external_htab == MMU_HASH64_KVM_MANAGED_HPT) {

        /*

         * HTAB is controlled by KVM. Fetch into temporary buffer

         */

        hptes = g_malloc(HASH_PTEG_SIZE_64);

        kvmppc_read_hptes(hptes, ptex, n);

    } else if (cpu->env.external_htab) {

        /*

         * HTAB is controlled by QEMU. Just point to the internally

         * accessible PTEG.

         */

        hptes = (ppc_hash_pte64_t *)(cpu->env.external_htab + pte_offset);

    } else if (cpu->env.htab_base) {

        hwaddr plen = n * HASH_PTE_SIZE_64;

        hptes = address_space_map(CPU(cpu)->as, cpu->env.htab_base + pte_offset,

                                 &plen, false);

        if (plen < (n * HASH_PTE_SIZE_64)) {

            hw_error("%s: Unable to map all requested HPTEs\n", __func__);

        }

    }

    return hptes;

}

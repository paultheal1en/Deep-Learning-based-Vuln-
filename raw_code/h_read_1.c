static target_ulong h_read(PowerPCCPU *cpu, sPAPREnvironment *spapr,

                           target_ulong opcode, target_ulong *args)

{

    CPUPPCState *env = &cpu->env;

    target_ulong flags = args[0];

    target_ulong pte_index = args[1];

    uint8_t *hpte;

    int i, ridx, n_entries = 1;



    if ((pte_index * HASH_PTE_SIZE_64) & ~env->htab_mask) {

        return H_PARAMETER;

    }



    if (flags & H_READ_4) {

        /* Clear the two low order bits */

        pte_index &= ~(3ULL);

        n_entries = 4;

    }



    hpte = env->external_htab + (pte_index * HASH_PTE_SIZE_64);



    for (i = 0, ridx = 0; i < n_entries; i++) {

        args[ridx++] = ldq_p(hpte);

        args[ridx++] = ldq_p(hpte + (HASH_PTE_SIZE_64/2));

        hpte += HASH_PTE_SIZE_64;

    }



    return H_SUCCESS;

}

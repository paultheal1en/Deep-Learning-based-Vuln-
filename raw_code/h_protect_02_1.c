static target_ulong h_protect(PowerPCCPU *cpu, sPAPREnvironment *spapr,

                              target_ulong opcode, target_ulong *args)

{

    CPUPPCState *env = &cpu->env;

    target_ulong flags = args[0];

    target_ulong pte_index = args[1];

    target_ulong avpn = args[2];

    hwaddr hpte;

    target_ulong v, r, rb;



    if ((pte_index * HASH_PTE_SIZE_64) & ~env->htab_mask) {

        return H_PARAMETER;

    }



    hpte = pte_index * HASH_PTE_SIZE_64;



    v = ppc_hash64_load_hpte0(env, hpte);

    r = ppc_hash64_load_hpte1(env, hpte);



    if ((v & HPTE64_V_VALID) == 0 ||

        ((flags & H_AVPN) && (v & ~0x7fULL) != avpn)) {

        return H_NOT_FOUND;

    }



    r &= ~(HPTE64_R_PP0 | HPTE64_R_PP | HPTE64_R_N |

           HPTE64_R_KEY_HI | HPTE64_R_KEY_LO);

    r |= (flags << 55) & HPTE64_R_PP0;

    r |= (flags << 48) & HPTE64_R_KEY_HI;

    r |= flags & (HPTE64_R_PP | HPTE64_R_N | HPTE64_R_KEY_LO);

    rb = compute_tlbie_rb(v, r, pte_index);

    ppc_hash64_store_hpte0(env, hpte, (v & ~HPTE64_V_VALID) | HPTE64_V_HPTE_DIRTY);

    ppc_tlb_invalidate_one(env, rb);

    ppc_hash64_store_hpte1(env, hpte, r);

    /* Don't need a memory barrier, due to qemu's global lock */

    ppc_hash64_store_hpte0(env, hpte, v | HPTE64_V_HPTE_DIRTY);

    return H_SUCCESS;

}

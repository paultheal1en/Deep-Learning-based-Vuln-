void r4k_helper_tlbp(CPUMIPSState *env)

{

    r4k_tlb_t *tlb;

    target_ulong mask;

    target_ulong tag;

    target_ulong VPN;

    uint8_t ASID;

    int i;



    ASID = env->CP0_EntryHi & 0xFF;

    for (i = 0; i < env->tlb->nb_tlb; i++) {

        tlb = &env->tlb->mmu.r4k.tlb[i];

        /* 1k pages are not supported. */

        mask = tlb->PageMask | ~(TARGET_PAGE_MASK << 1);

        tag = env->CP0_EntryHi & ~mask;

        VPN = tlb->VPN & ~mask;

#if defined(TARGET_MIPS64)

        tag &= env->SEGMask;

#endif

        /* Check ASID, virtual page number & size */

        if ((tlb->G == 1 || tlb->ASID == ASID) && VPN == tag) {

            /* TLB match */

            env->CP0_Index = i;

            break;

        }

    }

    if (i == env->tlb->nb_tlb) {

        /* No match.  Discard any shadow entries, if any of them match.  */

        for (i = env->tlb->nb_tlb; i < env->tlb->tlb_in_use; i++) {

            tlb = &env->tlb->mmu.r4k.tlb[i];

            /* 1k pages are not supported. */

            mask = tlb->PageMask | ~(TARGET_PAGE_MASK << 1);

            tag = env->CP0_EntryHi & ~mask;

            VPN = tlb->VPN & ~mask;

#if defined(TARGET_MIPS64)

            tag &= env->SEGMask;

#endif

            /* Check ASID, virtual page number & size */

            if ((tlb->G == 1 || tlb->ASID == ASID) && VPN == tag) {

                r4k_mips_tlb_flush_extra (env, i);

                break;

            }

        }



        env->CP0_Index |= 0x80000000;

    }

}

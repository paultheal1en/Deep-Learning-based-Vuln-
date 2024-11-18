static hwaddr ppc_hash64_htab_lookup(PowerPCCPU *cpu,

                                     ppc_slb_t *slb, target_ulong eaddr,

                                     ppc_hash_pte64_t *pte)

{

    CPUPPCState *env = &cpu->env;

    hwaddr pte_offset;

    hwaddr hash;

    uint64_t vsid, epnmask, epn, ptem;



    /* The SLB store path should prevent any bad page size encodings

     * getting in there, so: */

    assert(slb->sps);



    epnmask = ~((1ULL << slb->sps->page_shift) - 1);



    if (slb->vsid & SLB_VSID_B) {

        /* 1TB segment */

        vsid = (slb->vsid & SLB_VSID_VSID) >> SLB_VSID_SHIFT_1T;

        epn = (eaddr & ~SEGMENT_MASK_1T) & epnmask;

        hash = vsid ^ (vsid << 25) ^ (epn >> slb->sps->page_shift);

    } else {

        /* 256M segment */

        vsid = (slb->vsid & SLB_VSID_VSID) >> SLB_VSID_SHIFT;

        epn = (eaddr & ~SEGMENT_MASK_256M) & epnmask;

        hash = vsid ^ (epn >> slb->sps->page_shift);

    }

    ptem = (slb->vsid & SLB_VSID_PTEM) | ((epn >> 16) & HPTE64_V_AVPN);



    /* Page address translation */

    qemu_log_mask(CPU_LOG_MMU,

            "htab_base " TARGET_FMT_plx " htab_mask " TARGET_FMT_plx

            " hash " TARGET_FMT_plx "\n",

            env->htab_base, env->htab_mask, hash);



    /* Primary PTEG lookup */

    qemu_log_mask(CPU_LOG_MMU,

            "0 htab=" TARGET_FMT_plx "/" TARGET_FMT_plx

            " vsid=" TARGET_FMT_lx " ptem=" TARGET_FMT_lx

            " hash=" TARGET_FMT_plx "\n",

            env->htab_base, env->htab_mask, vsid, ptem,  hash);

    pte_offset = ppc_hash64_pteg_search(cpu, hash, slb, 0, ptem, pte);



    if (pte_offset == -1) {

        /* Secondary PTEG lookup */

        qemu_log_mask(CPU_LOG_MMU,

                "1 htab=" TARGET_FMT_plx "/" TARGET_FMT_plx

                " vsid=" TARGET_FMT_lx " api=" TARGET_FMT_lx

                " hash=" TARGET_FMT_plx "\n", env->htab_base,

                env->htab_mask, vsid, ptem, ~hash);



        pte_offset = ppc_hash64_pteg_search(cpu, ~hash, slb, 1, ptem, pte);

    }



    return pte_offset;

}

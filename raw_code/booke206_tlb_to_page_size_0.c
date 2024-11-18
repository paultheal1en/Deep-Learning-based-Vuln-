target_phys_addr_t booke206_tlb_to_page_size(CPUState *env, ppcmas_tlb_t *tlb)

{

    uint32_t tlbncfg;

    int tlbn = booke206_tlbm_to_tlbn(env, tlb);

    int tlbm_size;



    tlbncfg = env->spr[SPR_BOOKE_TLB0CFG + tlbn];



    if (tlbncfg & TLBnCFG_AVAIL) {

        tlbm_size = (tlb->mas1 & MAS1_TSIZE_MASK) >> MAS1_TSIZE_SHIFT;

    } else {

        tlbm_size = (tlbncfg & TLBnCFG_MINSIZE) >> TLBnCFG_MINSIZE_SHIFT;

        tlbm_size <<= 1;

    }



    return 1024ULL << tlbm_size;

}

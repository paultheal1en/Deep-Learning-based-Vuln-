int ppcmas_tlb_check(CPUState *env, ppcmas_tlb_t *tlb,

                     target_phys_addr_t *raddrp,

                     target_ulong address, uint32_t pid)

{

    target_ulong mask;

    uint32_t tlb_pid;



    /* Check valid flag */

    if (!(tlb->mas1 & MAS1_VALID)) {

        return -1;

    }



    mask = ~(booke206_tlb_to_page_size(env, tlb) - 1);

    LOG_SWTLB("%s: TLB ADDR=0x" TARGET_FMT_lx " PID=0x%x MAS1=0x%x MAS2=0x%"

              PRIx64 " mask=0x" TARGET_FMT_lx " MAS7_3=0x%" PRIx64 " MAS8=%x\n",

              __func__, address, pid, tlb->mas1, tlb->mas2, mask, tlb->mas7_3,

              tlb->mas8);



    /* Check PID */

    tlb_pid = (tlb->mas1 & MAS1_TID_MASK) >> MAS1_TID_SHIFT;

    if (tlb_pid != 0 && tlb_pid != pid) {

        return -1;

    }



    /* Check effective address */

    if ((address & mask) != (tlb->mas2 & MAS2_EPN_MASK)) {

        return -1;

    }

    *raddrp = (tlb->mas7_3 & mask) | (address & ~mask);



    return 0;

}

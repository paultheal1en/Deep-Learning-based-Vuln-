void HELPER(ipte)(CPUS390XState *env, uint64_t pto, uint64_t vaddr,

                  uint32_t m4)

{

    CPUState *cs = CPU(s390_env_get_cpu(env));

    uint64_t page = vaddr & TARGET_PAGE_MASK;

    uint64_t pte_addr, pte;



    /* Compute the page table entry address */

    pte_addr = (pto & _SEGMENT_ENTRY_ORIGIN);

    pte_addr += (vaddr & VADDR_PX) >> 9;



    /* Mark the page table entry as invalid */

    pte = ldq_phys(cs->as, pte_addr);

    pte |= _PAGE_INVALID;

    stq_phys(cs->as, pte_addr, pte);



    /* XXX we exploit the fact that Linux passes the exact virtual

       address here - it's not obliged to! */

    if (m4 & 1) {

        tlb_flush_page(cs, page);

    } else {

        tlb_flush_page_all_cpus_synced(cs, page);

    }



    /* XXX 31-bit hack */

    if (m4 & 1) {

        tlb_flush_page(cs, page ^ 0x80000000);

    } else {

        tlb_flush_page_all_cpus_synced(cs, page ^ 0x80000000);

    }

}

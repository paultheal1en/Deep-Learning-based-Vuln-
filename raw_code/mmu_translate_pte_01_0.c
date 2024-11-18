static int mmu_translate_pte(CPUS390XState *env, target_ulong vaddr,

                             uint64_t asc, uint64_t pt_entry,

                             target_ulong *raddr, int *flags, int rw, bool exc)

{

    if (pt_entry & _PAGE_INVALID) {

        DPRINTF("%s: PTE=0x%" PRIx64 " invalid\n", __func__, pt_entry);

        trigger_page_fault(env, vaddr, PGM_PAGE_TRANS, asc, rw, exc);

        return -1;

    }



    if (pt_entry & _PAGE_RO) {

        *flags &= ~PAGE_WRITE;

    }



    *raddr = pt_entry & _ASCE_ORIGIN;



    PTE_DPRINTF("%s: PTE=0x%" PRIx64 "\n", __func__, pt_entry);



    return 0;

}

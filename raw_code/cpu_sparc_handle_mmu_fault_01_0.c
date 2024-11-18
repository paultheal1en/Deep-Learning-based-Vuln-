int cpu_sparc_handle_mmu_fault (CPUState *env, target_ulong address, int rw,

                              int mmu_idx, int is_softmmu)

{

    target_ulong virt_addr, vaddr;

    target_phys_addr_t paddr;

    int error_code = 0, prot, ret = 0, access_index;



    error_code = get_physical_address(env, &paddr, &prot, &access_index,

                                      address, rw, mmu_idx);

    if (error_code == 0) {

        virt_addr = address & TARGET_PAGE_MASK;

        vaddr = virt_addr + ((address & TARGET_PAGE_MASK) &

                             (TARGET_PAGE_SIZE - 1));

#ifdef DEBUG_MMU

        printf("Translate at 0x%" PRIx64 " -> 0x%" PRIx64 ", vaddr 0x%" PRIx64

               "\n", address, paddr, vaddr);

#endif

        ret = tlb_set_page_exec(env, vaddr, paddr, prot, mmu_idx, is_softmmu);

        return ret;

    }

    // XXX

    return 1;

}

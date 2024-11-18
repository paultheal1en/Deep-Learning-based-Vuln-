tb_page_addr_t get_page_addr_code(CPUArchState *env1, target_ulong addr)

{

    int mmu_idx, page_index, pd;

    void *p;

    MemoryRegion *mr;



    page_index = (addr >> TARGET_PAGE_BITS) & (CPU_TLB_SIZE - 1);

    mmu_idx = cpu_mmu_index(env1);

    if (unlikely(env1->tlb_table[mmu_idx][page_index].addr_code !=

                 (addr & TARGET_PAGE_MASK))) {

#ifdef CONFIG_TCG_PASS_AREG0

        cpu_ldub_code(env1, addr);

#else

        ldub_code(addr);

#endif

    }

    pd = env1->iotlb[mmu_idx][page_index] & ~TARGET_PAGE_MASK;

    mr = iotlb_to_region(pd);

    if (mr != &io_mem_ram && mr != &io_mem_rom

        && mr != &io_mem_notdirty && !mr->rom_device) {

#if defined(TARGET_ALPHA) || defined(TARGET_MIPS) || defined(TARGET_SPARC)

        cpu_unassigned_access(env1, addr, 0, 1, 0, 4);

#else

        cpu_abort(env1, "Trying to execute code outside RAM or ROM at 0x" TARGET_FMT_lx "\n", addr);

#endif

    }

    p = (void *)((uintptr_t)addr + env1->tlb_table[mmu_idx][page_index].addend);

    return qemu_ram_addr_from_host_nofail(p);

}

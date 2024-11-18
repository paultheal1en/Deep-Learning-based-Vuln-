int cpu_m68k_handle_mmu_fault (CPUState *env, target_ulong address, int rw,

                               int mmu_idx, int is_softmmu)

{

    int prot;



    address &= TARGET_PAGE_MASK;

    prot = PAGE_READ | PAGE_WRITE;

    return tlb_set_page(env, address, address, prot, mmu_idx, is_softmmu);

}

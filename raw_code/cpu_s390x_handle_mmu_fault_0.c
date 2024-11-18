int cpu_s390x_handle_mmu_fault (CPUState *env, target_ulong address, int rw,

                                int mmu_idx, int is_softmmu)

{

    target_ulong phys;

    int prot;



    /* XXX: implement mmu */



    phys = address;

    prot = PAGE_READ | PAGE_WRITE;



    return tlb_set_page(env, address & TARGET_PAGE_MASK,

                        phys & TARGET_PAGE_MASK, prot,

                        mmu_idx, is_softmmu);

}

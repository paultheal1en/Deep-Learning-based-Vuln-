int mmu_translate_real(CPUS390XState *env, target_ulong raddr, int rw,

                       target_ulong *addr, int *flags)

{

    /* TODO: low address protection once we flush the tlb on cr changes */

    *flags = PAGE_READ | PAGE_WRITE;

    *addr = mmu_real2abs(env, raddr);



    /* TODO: storage key handling */

    return 0;

}

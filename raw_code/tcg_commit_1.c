static void tcg_commit(MemoryListener *listener)

{

    CPUAddressSpace *cpuas;

    AddressSpaceDispatch *d;



    /* since each CPU stores ram addresses in its TLB cache, we must

       reset the modified entries */

    cpuas = container_of(listener, CPUAddressSpace, tcg_as_listener);

    cpu_reloading_memory_map();

    /* The CPU and TLB are protected by the iothread lock.

     * We reload the dispatch pointer now because cpu_reloading_memory_map()

     * may have split the RCU critical section.

     */

    d = atomic_rcu_read(&cpuas->as->dispatch);

    cpuas->memory_dispatch = d;

    tlb_flush(cpuas->cpu, 1);

}

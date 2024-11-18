void *qemu_anon_ram_alloc(size_t size)

{

    void *ptr;



    /* FIXME: this is not exactly optimal solution since VirtualAlloc

       has 64Kb granularity, but at least it guarantees us that the

       memory is page aligned. */

    ptr = VirtualAlloc(NULL, size, MEM_COMMIT, PAGE_READWRITE);

    trace_qemu_anon_ram_alloc(size, ptr);

    return ptr;

}

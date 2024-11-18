void *qemu_vmalloc(size_t size)

{

    /* FIXME: this is not exactly optimal solution since VirtualAlloc

       has 64Kb granularity, but at least it guarantees us that the

       memory is page aligned. */

    if (!size) {

        abort();

    }

    return oom_check(VirtualAlloc(NULL, size, MEM_COMMIT, PAGE_READWRITE));

}

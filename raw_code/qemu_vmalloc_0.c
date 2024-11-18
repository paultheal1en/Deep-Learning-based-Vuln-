void *qemu_vmalloc(size_t size)

{

    void *p;

    unsigned long addr;

    mmap_lock();

    /* Use map and mark the pages as used.  */

    p = mmap(NULL, size, PROT_READ | PROT_WRITE,

             MAP_PRIVATE | MAP_ANON, -1, 0);



    addr = (unsigned long)p;

    if (addr == (target_ulong) addr) {

        /* Allocated region overlaps guest address space.

           This may recurse.  */

        page_set_flags(addr & TARGET_PAGE_MASK, TARGET_PAGE_ALIGN(addr + size),

                       PAGE_RESERVED);

    }



    mmap_unlock();

    return p;

}

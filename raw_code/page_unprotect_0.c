int page_unprotect(target_ulong address, unsigned long pc, void *puc)

{

    unsigned int page_index, prot, pindex;

    PageDesc *p, *p1;

    target_ulong host_start, host_end, addr;



    /* Technically this isn't safe inside a signal handler.  However we

       know this only ever happens in a synchronous SEGV handler, so in

       practice it seems to be ok.  */

    mmap_lock();



    host_start = address & qemu_host_page_mask;

    page_index = host_start >> TARGET_PAGE_BITS;

    p1 = page_find(page_index);

    if (!p1) {

        mmap_unlock();

        return 0;

    }

    host_end = host_start + qemu_host_page_size;

    p = p1;

    prot = 0;

    for(addr = host_start;addr < host_end; addr += TARGET_PAGE_SIZE) {

        prot |= p->flags;

        p++;

    }

    /* if the page was really writable, then we change its

       protection back to writable */

    if (prot & PAGE_WRITE_ORG) {

        pindex = (address - host_start) >> TARGET_PAGE_BITS;

        if (!(p1[pindex].flags & PAGE_WRITE)) {

            mprotect((void *)g2h(host_start), qemu_host_page_size,

                     (prot & PAGE_BITS) | PAGE_WRITE);

            p1[pindex].flags |= PAGE_WRITE;

            /* and since the content will be modified, we must invalidate

               the corresponding translated code. */

            tb_invalidate_phys_page(address, pc, puc);

#ifdef DEBUG_TB_CHECK

            tb_invalidate_check(address);

#endif

            mmap_unlock();

            return 1;

        }

    }

    mmap_unlock();

    return 0;

}

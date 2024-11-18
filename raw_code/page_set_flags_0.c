void page_set_flags(target_ulong start, target_ulong end, int flags)

{

    PageDesc *p;

    target_ulong addr;



    /* mmap_lock should already be held.  */

    start = start & TARGET_PAGE_MASK;

    end = TARGET_PAGE_ALIGN(end);

    if (flags & PAGE_WRITE)

        flags |= PAGE_WRITE_ORG;

    for(addr = start; addr < end; addr += TARGET_PAGE_SIZE) {

        p = page_find_alloc(addr >> TARGET_PAGE_BITS);

        /* We may be called for host regions that are outside guest

           address space.  */

        if (!p)

            return;

        /* if the write protection is set, then we invalidate the code

           inside */

        if (!(p->flags & PAGE_WRITE) &&

            (flags & PAGE_WRITE) &&

            p->first_tb) {

            tb_invalidate_phys_page(addr, 0, NULL);

        }

        p->flags = flags;

    }

}

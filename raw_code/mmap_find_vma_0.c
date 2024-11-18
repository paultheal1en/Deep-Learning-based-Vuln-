abi_ulong mmap_find_vma(abi_ulong start, abi_ulong size)

{

    void *ptr, *prev;

    abi_ulong addr;

    int wrapped, repeat;



    /* If 'start' == 0, then a default start address is used. */

    if (start == 0) {

        start = mmap_next_start;

    } else {

        start &= qemu_host_page_mask;

    }



    size = HOST_PAGE_ALIGN(size);



    if (RESERVED_VA) {

        return mmap_find_vma_reserved(start, size);

    }



    addr = start;

    wrapped = repeat = 0;

    prev = 0;



    for (;; prev = ptr) {

        /*

         * Reserve needed memory area to avoid a race.

         * It should be discarded using:

         *  - mmap() with MAP_FIXED flag

         *  - mremap() with MREMAP_FIXED flag

         *  - shmat() with SHM_REMAP flag

         */

        ptr = mmap(g2h(addr), size, PROT_NONE,

                   MAP_ANONYMOUS|MAP_PRIVATE|MAP_NORESERVE, -1, 0);



        /* ENOMEM, if host address space has no memory */

        if (ptr == MAP_FAILED) {

            return (abi_ulong)-1;

        }



        /* Count the number of sequential returns of the same address.

           This is used to modify the search algorithm below.  */

        repeat = (ptr == prev ? repeat + 1 : 0);



        if (h2g_valid(ptr + size - 1)) {

            addr = h2g(ptr);



            if ((addr & ~TARGET_PAGE_MASK) == 0) {

                /* Success.  */

                if (start == mmap_next_start && addr >= TASK_UNMAPPED_BASE) {

                    mmap_next_start = addr + size;

                }

                return addr;

            }



            /* The address is not properly aligned for the target.  */

            switch (repeat) {

            case 0:

                /* Assume the result that the kernel gave us is the

                   first with enough free space, so start again at the

                   next higher target page.  */

                addr = TARGET_PAGE_ALIGN(addr);

                break;

            case 1:

                /* Sometimes the kernel decides to perform the allocation

                   at the top end of memory instead.  */

                addr &= TARGET_PAGE_MASK;

                break;

            case 2:

                /* Start over at low memory.  */

                addr = 0;

                break;

            default:

                /* Fail.  This unaligned block must the last.  */

                addr = -1;

                break;

            }

        } else {

            /* Since the result the kernel gave didn't fit, start

               again at low memory.  If any repetition, fail.  */

            addr = (repeat ? -1 : 0);

        }



        /* Unmap and try again.  */

        munmap(ptr, size);



        /* ENOMEM if we checked the whole of the target address space.  */

        if (addr == -1ul) {

            return (abi_ulong)-1;

        } else if (addr == 0) {

            if (wrapped) {

                return (abi_ulong)-1;

            }

            wrapped = 1;

            /* Don't actually use 0 when wrapping, instead indicate

               that we'd truely like an allocation in low memory.  */

            addr = (mmap_min_addr > TARGET_PAGE_SIZE

                     ? TARGET_PAGE_ALIGN(mmap_min_addr)

                     : TARGET_PAGE_SIZE);

        } else if (wrapped && addr >= start) {

            return (abi_ulong)-1;

        }

    }

}

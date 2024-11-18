unsigned long init_guest_space(unsigned long host_start,

                               unsigned long host_size,

                               unsigned long guest_start,

                               bool fixed)

{

    unsigned long current_start, real_start;

    int flags;



    assert(host_start || host_size);



    /* If just a starting address is given, then just verify that

     * address.  */

    if (host_start && !host_size) {

        if (guest_validate_base(host_start)) {

            return host_start;

        } else {

            return (unsigned long)-1;

        }

    }



    /* Setup the initial flags and start address.  */

    current_start = host_start & qemu_host_page_mask;

    flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE;

    if (fixed) {

        flags |= MAP_FIXED;

    }



    /* Otherwise, a non-zero size region of memory needs to be mapped

     * and validated.  */

    while (1) {

        /* Do not use mmap_find_vma here because that is limited to the

         * guest address space.  We are going to make the

         * guest address space fit whatever we're given.

         */

        real_start = (unsigned long)

            mmap((void *)current_start, host_size, PROT_NONE, flags, -1, 0);

        if (real_start == (unsigned long)-1) {

            return (unsigned long)-1;

        }



        if ((real_start == current_start)

            && guest_validate_base(real_start - guest_start)) {

            break;

        }



        /* That address didn't work.  Unmap and try a different one.

         * The address the host picked because is typically right at

         * the top of the host address space and leaves the guest with

         * no usable address space.  Resort to a linear search.  We

         * already compensated for mmap_min_addr, so this should not

         * happen often.  Probably means we got unlucky and host

         * address space randomization put a shared library somewhere

         * inconvenient.

         */

        munmap((void *)real_start, host_size);

        current_start += qemu_host_page_size;

        if (host_start == current_start) {

            /* Theoretically possible if host doesn't have any suitably

             * aligned areas.  Normally the first mmap will fail.

             */

            return (unsigned long)-1;

        }

    }



    return real_start;

}

static void zero_bss(abi_ulong elf_bss, abi_ulong last_bss, int prot)

{

    uintptr_t host_start, host_map_start, host_end;



    last_bss = TARGET_PAGE_ALIGN(last_bss);



    /* ??? There is confusion between qemu_real_host_page_size and

       qemu_host_page_size here and elsewhere in target_mmap, which

       may lead to the end of the data section mapping from the file

       not being mapped.  At least there was an explicit test and

       comment for that here, suggesting that "the file size must

       be known".  The comment probably pre-dates the introduction

       of the fstat system call in target_mmap which does in fact

       find out the size.  What isn't clear is if the workaround

       here is still actually needed.  For now, continue with it,

       but merge it with the "normal" mmap that would allocate the bss.  */



    host_start = (uintptr_t) g2h(elf_bss);

    host_end = (uintptr_t) g2h(last_bss);

    host_map_start = (host_start + qemu_real_host_page_size - 1);

    host_map_start &= -qemu_real_host_page_size;



    if (host_map_start < host_end) {

        void *p = mmap((void *)host_map_start, host_end - host_map_start,

                       prot, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (p == MAP_FAILED) {

            perror("cannot mmap brk");

            exit(-1);

        }



        /* Since we didn't use target_mmap, make sure to record

           the validity of the pages with qemu.  */

        page_set_flags(elf_bss & TARGET_PAGE_MASK, last_bss, prot|PAGE_VALID);

    }



    if (host_start < host_map_start) {

        memset((void *)host_start, 0, host_map_start - host_start);

    }

}

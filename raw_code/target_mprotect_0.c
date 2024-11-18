int target_mprotect(abi_ulong start, abi_ulong len, int prot)

{

    abi_ulong end, host_start, host_end, addr;

    int prot1, ret;



#ifdef DEBUG_MMAP

    printf("mprotect: start=0x" TARGET_FMT_lx

           "len=0x" TARGET_FMT_lx " prot=%c%c%c\n", start, len,

           prot & PROT_READ ? 'r' : '-',

           prot & PROT_WRITE ? 'w' : '-',

           prot & PROT_EXEC ? 'x' : '-');

#endif



    if ((start & ~TARGET_PAGE_MASK) != 0)

        return -EINVAL;

    len = TARGET_PAGE_ALIGN(len);

    end = start + len;

    if (end < start)

        return -EINVAL;

    if (prot & ~(PROT_READ | PROT_WRITE | PROT_EXEC))

        return -EINVAL;

    if (len == 0)

        return 0;



    host_start = start & qemu_host_page_mask;

    host_end = HOST_PAGE_ALIGN(end);

    if (start > host_start) {

        /* handle host page containing start */

        prot1 = prot;

        for(addr = host_start; addr < start; addr += TARGET_PAGE_SIZE) {

            prot1 |= page_get_flags(addr);

        }

        if (host_end == host_start + qemu_host_page_size) {

            for(addr = end; addr < host_end; addr += TARGET_PAGE_SIZE) {

                prot1 |= page_get_flags(addr);

            }

            end = host_end;

        }

        ret = mprotect(g2h(host_start), qemu_host_page_size, prot1 & PAGE_BITS);

        if (ret != 0)

            return ret;

        host_start += qemu_host_page_size;

    }

    if (end < host_end) {

        prot1 = prot;

        for(addr = end; addr < host_end; addr += TARGET_PAGE_SIZE) {

            prot1 |= page_get_flags(addr);

        }

        ret = mprotect(g2h(host_end - qemu_host_page_size), qemu_host_page_size,

                       prot1 & PAGE_BITS);

        if (ret != 0)

            return ret;

        host_end -= qemu_host_page_size;

    }



    /* handle the pages in the middle */

    if (host_start < host_end) {

        ret = mprotect(g2h(host_start), host_end - host_start, prot);

        if (ret != 0)

            return ret;

    }

    page_set_flags(start, start + len, prot | PAGE_VALID);

    return 0;

}

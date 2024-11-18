abi_long target_mmap(abi_ulong start, abi_ulong len, int prot,

                     int flags, int fd, abi_ulong offset)

{

    abi_ulong ret, end, real_start, real_end, retaddr, host_offset, host_len;



    mmap_lock();

#ifdef DEBUG_MMAP

    {

        printf("mmap: start=0x" TARGET_ABI_FMT_lx

               " len=0x" TARGET_ABI_FMT_lx " prot=%c%c%c flags=",

               start, len,

               prot & PROT_READ ? 'r' : '-',

               prot & PROT_WRITE ? 'w' : '-',

               prot & PROT_EXEC ? 'x' : '-');

        if (flags & MAP_FIXED)

            printf("MAP_FIXED ");

        if (flags & MAP_ANONYMOUS)

            printf("MAP_ANON ");

        switch(flags & MAP_TYPE) {

        case MAP_PRIVATE:

            printf("MAP_PRIVATE ");

            break;

        case MAP_SHARED:

            printf("MAP_SHARED ");

            break;

        default:

            printf("[MAP_TYPE=0x%x] ", flags & MAP_TYPE);

            break;

        }

        printf("fd=%d offset=" TARGET_ABI_FMT_lx "\n", fd, offset);

    }

#endif



    if (offset & ~TARGET_PAGE_MASK) {

        errno = EINVAL;

        goto fail;

    }



    len = TARGET_PAGE_ALIGN(len);

    if (len == 0)

        goto the_end;

    real_start = start & qemu_host_page_mask;

    host_offset = offset & qemu_host_page_mask;



    /* If the user is asking for the kernel to find a location, do that

       before we truncate the length for mapping files below.  */

    if (!(flags & MAP_FIXED)) {

        host_len = len + offset - host_offset;

        host_len = HOST_PAGE_ALIGN(host_len);

        start = mmap_find_vma(real_start, host_len);

        if (start == (abi_ulong)-1) {

            errno = ENOMEM;

            goto fail;

        }

    }



    /* When mapping files into a memory area larger than the file, accesses

       to pages beyond the file size will cause a SIGBUS. 



       For example, if mmaping a file of 100 bytes on a host with 4K pages

       emulating a target with 8K pages, the target expects to be able to

       access the first 8K. But the host will trap us on any access beyond

       4K.  



       When emulating a target with a larger page-size than the hosts, we

       may need to truncate file maps at EOF and add extra anonymous pages

       up to the targets page boundary.  */



    if ((qemu_real_host_page_size < TARGET_PAGE_SIZE)

        && !(flags & MAP_ANONYMOUS)) {

       struct stat sb;



       if (fstat (fd, &sb) == -1)

           goto fail;



       /* Are we trying to create a map beyond EOF?.  */

       if (offset + len > sb.st_size) {

           /* If so, truncate the file map at eof aligned with 

              the hosts real pagesize. Additional anonymous maps

              will be created beyond EOF.  */

           len = REAL_HOST_PAGE_ALIGN(sb.st_size - offset);

       }

    }



    if (!(flags & MAP_FIXED)) {

        unsigned long host_start;

        void *p;



        host_len = len + offset - host_offset;

        host_len = HOST_PAGE_ALIGN(host_len);



        /* Note: we prefer to control the mapping address. It is

           especially important if qemu_host_page_size >

           qemu_real_host_page_size */

        p = mmap(g2h(start), host_len, prot,

                 flags | MAP_FIXED | MAP_ANONYMOUS, -1, 0);

        if (p == MAP_FAILED)

            goto fail;

        /* update start so that it points to the file position at 'offset' */

        host_start = (unsigned long)p;

        if (!(flags & MAP_ANONYMOUS)) {

            p = mmap(g2h(start), len, prot,

                     flags | MAP_FIXED, fd, host_offset);

            if (p == MAP_FAILED) {

                munmap(g2h(start), host_len);

                goto fail;

            }

            host_start += offset - host_offset;

        }

        start = h2g(host_start);

    } else {

        if (start & ~TARGET_PAGE_MASK) {

            errno = EINVAL;

            goto fail;

        }

        end = start + len;

        real_end = HOST_PAGE_ALIGN(end);



	/*

	 * Test if requested memory area fits target address space

	 * It can fail only on 64-bit host with 32-bit target.

	 * On any other target/host host mmap() handles this error correctly.

	 */

        if ((unsigned long)start + len - 1 > (abi_ulong) -1) {

            errno = EINVAL;

            goto fail;

        }



        /* worst case: we cannot map the file because the offset is not

           aligned, so we read it */

        if (!(flags & MAP_ANONYMOUS) &&

            (offset & ~qemu_host_page_mask) != (start & ~qemu_host_page_mask)) {

            /* msync() won't work here, so we return an error if write is

               possible while it is a shared mapping */

            if ((flags & MAP_TYPE) == MAP_SHARED &&

                (prot & PROT_WRITE)) {

                errno = EINVAL;

                goto fail;

            }

            retaddr = target_mmap(start, len, prot | PROT_WRITE,

                                  MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS,

                                  -1, 0);

            if (retaddr == -1)

                goto fail;

            if (pread(fd, g2h(start), len, offset) == -1)

                goto fail;

            if (!(prot & PROT_WRITE)) {

                ret = target_mprotect(start, len, prot);

                assert(ret == 0);

            }

            goto the_end;

        }

        

        /* handle the start of the mapping */

        if (start > real_start) {

            if (real_end == real_start + qemu_host_page_size) {

                /* one single host page */

                ret = mmap_frag(real_start, start, end,

                                prot, flags, fd, offset);

                if (ret == -1)

                    goto fail;

                goto the_end1;

            }

            ret = mmap_frag(real_start, start, real_start + qemu_host_page_size,

                            prot, flags, fd, offset);

            if (ret == -1)

                goto fail;

            real_start += qemu_host_page_size;

        }

        /* handle the end of the mapping */

        if (end < real_end) {

            ret = mmap_frag(real_end - qemu_host_page_size,

                            real_end - qemu_host_page_size, end,

                            prot, flags, fd,

                            offset + real_end - qemu_host_page_size - start);

            if (ret == -1)

                goto fail;

            real_end -= qemu_host_page_size;

        }



        /* map the middle (easier) */

        if (real_start < real_end) {

            void *p;

            unsigned long offset1;

            if (flags & MAP_ANONYMOUS)

                offset1 = 0;

            else

                offset1 = offset + real_start - start;

            p = mmap(g2h(real_start), real_end - real_start,

                     prot, flags, fd, offset1);

            if (p == MAP_FAILED)

                goto fail;

        }

    }

 the_end1:

    page_set_flags(start, start + len, prot | PAGE_VALID);

 the_end:

#ifdef DEBUG_MMAP

    printf("ret=0x" TARGET_ABI_FMT_lx "\n", start);

    page_dump(stdout);

    printf("\n");

#endif

    tb_invalidate_phys_range(start, start + len);

    mmap_unlock();

    return start;

fail:

    mmap_unlock();

    return -1;

}

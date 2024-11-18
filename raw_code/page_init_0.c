static void page_init(void)

{

    /* NOTE: we can always suppose that qemu_host_page_size >=

       TARGET_PAGE_SIZE */

#ifdef _WIN32

    {

        SYSTEM_INFO system_info;



        GetSystemInfo(&system_info);

        qemu_real_host_page_size = system_info.dwPageSize;

    }

#else

    qemu_real_host_page_size = getpagesize();

#endif

    if (qemu_host_page_size == 0) {

        qemu_host_page_size = qemu_real_host_page_size;

    }

    if (qemu_host_page_size < TARGET_PAGE_SIZE) {

        qemu_host_page_size = TARGET_PAGE_SIZE;

    }

    qemu_host_page_mask = ~(qemu_host_page_size - 1);



#if defined(CONFIG_BSD) && defined(CONFIG_USER_ONLY)

    {

#ifdef HAVE_KINFO_GETVMMAP

        struct kinfo_vmentry *freep;

        int i, cnt;



        freep = kinfo_getvmmap(getpid(), &cnt);

        if (freep) {

            mmap_lock();

            for (i = 0; i < cnt; i++) {

                unsigned long startaddr, endaddr;



                startaddr = freep[i].kve_start;

                endaddr = freep[i].kve_end;

                if (h2g_valid(startaddr)) {

                    startaddr = h2g(startaddr) & TARGET_PAGE_MASK;



                    if (h2g_valid(endaddr)) {

                        endaddr = h2g(endaddr);

                        page_set_flags(startaddr, endaddr, PAGE_RESERVED);

                    } else {

#if TARGET_ABI_BITS <= L1_MAP_ADDR_SPACE_BITS

                        endaddr = ~0ul;

                        page_set_flags(startaddr, endaddr, PAGE_RESERVED);

#endif

                    }

                }

            }

            free(freep);

            mmap_unlock();

        }

#else

        FILE *f;



        last_brk = (unsigned long)sbrk(0);



        f = fopen("/compat/linux/proc/self/maps", "r");

        if (f) {

            mmap_lock();



            do {

                unsigned long startaddr, endaddr;

                int n;



                n = fscanf(f, "%lx-%lx %*[^\n]\n", &startaddr, &endaddr);



                if (n == 2 && h2g_valid(startaddr)) {

                    startaddr = h2g(startaddr) & TARGET_PAGE_MASK;



                    if (h2g_valid(endaddr)) {

                        endaddr = h2g(endaddr);

                    } else {

                        endaddr = ~0ul;

                    }

                    page_set_flags(startaddr, endaddr, PAGE_RESERVED);

                }

            } while (!feof(f));



            fclose(f);

            mmap_unlock();

        }

#endif

    }

#endif

}

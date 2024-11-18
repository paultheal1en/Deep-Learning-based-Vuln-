void kqemu_modify_page(CPUState *env, ram_addr_t ram_addr)

{

    unsigned long page_index;

    int ret;

#ifdef _WIN32

    DWORD temp;

#endif



    page_index = ram_addr >> TARGET_PAGE_BITS;

    if (!modified_ram_pages_table[page_index]) {

#if 0

        printf("%d: modify_page=%08lx\n", nb_modified_ram_pages, ram_addr);

#endif

        modified_ram_pages_table[page_index] = 1;

        modified_ram_pages[nb_modified_ram_pages++] = ram_addr;

        if (nb_modified_ram_pages >= KQEMU_MAX_MODIFIED_RAM_PAGES) {

            /* flush */

#ifdef _WIN32

            ret = DeviceIoControl(kqemu_fd, KQEMU_MODIFY_RAM_PAGES,

                                  &nb_modified_ram_pages,

                                  sizeof(nb_modified_ram_pages),

                                  NULL, 0, &temp, NULL);

#else

            ret = ioctl(kqemu_fd, KQEMU_MODIFY_RAM_PAGES,

                        &nb_modified_ram_pages);

#endif

            kqemu_reset_modified_ram_pages();

        }

    }

}

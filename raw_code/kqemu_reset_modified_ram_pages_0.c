static void kqemu_reset_modified_ram_pages(void)

{

    int i;

    unsigned long page_index;



    for(i = 0; i < nb_modified_ram_pages; i++) {

        page_index = modified_ram_pages[i] >> TARGET_PAGE_BITS;

        modified_ram_pages_table[page_index] = 0;

    }

    nb_modified_ram_pages = 0;

}

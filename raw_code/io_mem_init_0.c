static void io_mem_init(void)

{

    int i;



    cpu_register_io_memory_fixed(IO_MEM_ROM, error_mem_read, unassigned_mem_write, NULL);

    cpu_register_io_memory_fixed(IO_MEM_UNASSIGNED, unassigned_mem_read, unassigned_mem_write, NULL);

    cpu_register_io_memory_fixed(IO_MEM_NOTDIRTY, error_mem_read, notdirty_mem_write, NULL);

    for (i=0; i<5; i++)

        io_mem_used[i] = 1;



    io_mem_watch = cpu_register_io_memory(watch_mem_read,

                                          watch_mem_write, NULL);

#ifdef CONFIG_KQEMU

    if (kqemu_phys_ram_base) {

        /* alloc dirty bits array */

        phys_ram_dirty = qemu_vmalloc(kqemu_phys_ram_size >> TARGET_PAGE_BITS);

        memset(phys_ram_dirty, 0xff, kqemu_phys_ram_size >> TARGET_PAGE_BITS);

    }

#endif

}

static void puv3_board_init(CPUUniCore32State *env, ram_addr_t ram_size)

{

    MemoryRegion *ram_memory = g_new(MemoryRegion, 1);



    /* SDRAM at address zero.  */

    memory_region_init_ram(ram_memory, NULL, "puv3.ram", ram_size,

                           &error_abort);

    vmstate_register_ram_global(ram_memory);

    memory_region_add_subregion(get_system_memory(), 0, ram_memory);

}

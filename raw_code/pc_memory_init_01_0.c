FWCfgState *pc_memory_init(MemoryRegion *system_memory,

                           const char *kernel_filename,

                           const char *kernel_cmdline,

                           const char *initrd_filename,

                           ram_addr_t below_4g_mem_size,

                           ram_addr_t above_4g_mem_size,

                           MemoryRegion *rom_memory,

                           MemoryRegion **ram_memory,

                           PcGuestInfo *guest_info)

{

    int linux_boot, i;

    MemoryRegion *ram, *option_rom_mr;

    MemoryRegion *ram_below_4g, *ram_above_4g;

    FWCfgState *fw_cfg;



    linux_boot = (kernel_filename != NULL);



    /* Allocate RAM.  We allocate it as a single memory region and use

     * aliases to address portions of it, mostly for backwards compatibility

     * with older qemus that used qemu_ram_alloc().

     */

    ram = g_malloc(sizeof(*ram));

    memory_region_init_ram(ram, NULL, "pc.ram",

                           below_4g_mem_size + above_4g_mem_size);

    vmstate_register_ram_global(ram);

    *ram_memory = ram;

    ram_below_4g = g_malloc(sizeof(*ram_below_4g));

    memory_region_init_alias(ram_below_4g, NULL, "ram-below-4g", ram,

                             0, below_4g_mem_size);

    memory_region_add_subregion(system_memory, 0, ram_below_4g);

    e820_add_entry(0, below_4g_mem_size, E820_RAM);

    if (above_4g_mem_size > 0) {

        ram_above_4g = g_malloc(sizeof(*ram_above_4g));

        memory_region_init_alias(ram_above_4g, NULL, "ram-above-4g", ram,

                                 below_4g_mem_size, above_4g_mem_size);

        memory_region_add_subregion(system_memory, 0x100000000ULL,

                                    ram_above_4g);

        e820_add_entry(0x100000000ULL, above_4g_mem_size, E820_RAM);

    }





    /* Initialize PC system firmware */

    pc_system_firmware_init(rom_memory, guest_info->isapc_ram_fw);



    option_rom_mr = g_malloc(sizeof(*option_rom_mr));

    memory_region_init_ram(option_rom_mr, NULL, "pc.rom", PC_ROM_SIZE);

    vmstate_register_ram_global(option_rom_mr);

    memory_region_add_subregion_overlap(rom_memory,

                                        PC_ROM_MIN_VGA,

                                        option_rom_mr,

                                        1);



    fw_cfg = bochs_bios_init();

    rom_set_fw(fw_cfg);



    if (linux_boot) {

        load_linux(fw_cfg, kernel_filename, initrd_filename, kernel_cmdline, below_4g_mem_size);

    }



    for (i = 0; i < nb_option_roms; i++) {

        rom_add_option(option_rom[i].name, option_rom[i].bootindex);

    }

    guest_info->fw_cfg = fw_cfg;

    return fw_cfg;

}

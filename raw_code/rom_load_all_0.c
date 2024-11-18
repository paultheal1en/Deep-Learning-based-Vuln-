int rom_load_all(void)

{

    target_phys_addr_t addr = 0;

    MemoryRegionSection section;

    Rom *rom;



    QTAILQ_FOREACH(rom, &roms, next) {

        if (rom->fw_file) {

            continue;

        }

        if (addr > rom->addr) {

            fprintf(stderr, "rom: requested regions overlap "

                    "(rom %s. free=0x" TARGET_FMT_plx

                    ", addr=0x" TARGET_FMT_plx ")\n",

                    rom->name, addr, rom->addr);

            return -1;

        }

        addr  = rom->addr;

        addr += rom->romsize;

        section = memory_region_find(get_system_memory(), rom->addr, 1);

        rom->isrom = section.size && memory_region_is_rom(section.mr);

    }

    qemu_register_reset(rom_reset, NULL);

    roms_loaded = 1;

    return 0;

}

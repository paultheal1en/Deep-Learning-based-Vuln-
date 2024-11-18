int rom_load_all(void)

{

    target_phys_addr_t addr = 0;

    int memtype;

    Rom *rom;



    QTAILQ_FOREACH(rom, &roms, next) {

        if (addr < rom->min)

            addr = rom->min;

        if (rom->max) {

            /* load address range */

            if (rom->align) {

                addr += (rom->align-1);

                addr &= ~(rom->align-1);

            }

            if (addr + rom->romsize > rom->max) {

                fprintf(stderr, "rom: out of memory (rom %s, "

                        "addr 0x" TARGET_FMT_plx

                        ", size 0x%zx, max 0x" TARGET_FMT_plx ")\n",

                        rom->name, addr, rom->romsize, rom->max);

                return -1;

            }

        } else {

            /* fixed address requested */

            if (addr != rom->min) {

                fprintf(stderr, "rom: requested regions overlap "

                        "(rom %s. free=0x" TARGET_FMT_plx

                        ", addr=0x" TARGET_FMT_plx ")\n",

                        rom->name, addr, rom->min);

                return -1;

            }

        }

        rom->addr = addr;

        addr += rom->romsize;

        memtype = cpu_get_physical_page_desc(rom->addr) & (3 << IO_MEM_SHIFT);

        if (memtype == IO_MEM_ROM)

            rom->isrom = 1;

    }

    qemu_register_reset(rom_reset, NULL);

    roms_loaded = 1;

    return 0;

}

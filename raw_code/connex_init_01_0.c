static void connex_init(MachineState *machine)

{

    PXA2xxState *cpu;

    DriveInfo *dinfo;

    int be;

    MemoryRegion *address_space_mem = get_system_memory();



    uint32_t connex_rom = 0x01000000;

    uint32_t connex_ram = 0x04000000;



    cpu = pxa255_init(address_space_mem, connex_ram);



    dinfo = drive_get(IF_PFLASH, 0, 0);

    if (!dinfo && !qtest_enabled()) {

        fprintf(stderr, "A flash image must be given with the "

                "'pflash' parameter\n");

        exit(1);

    }



#ifdef TARGET_WORDS_BIGENDIAN

    be = 1;

#else

    be = 0;

#endif

    if (!pflash_cfi01_register(0x00000000, NULL, "connext.rom", connex_rom,

                               dinfo ? blk_bs(blk_by_legacy_dinfo(dinfo)) : NULL,

                               sector_len, connex_rom / sector_len,

                               2, 0, 0, 0, 0, be)) {

        fprintf(stderr, "qemu: Error registering flash memory.\n");

        exit(1);

    }



    /* Interrupt line of NIC is connected to GPIO line 36 */

    smc91c111_init(&nd_table[0], 0x04000300,

                    qdev_get_gpio_in(cpu->gpio, 36));

}

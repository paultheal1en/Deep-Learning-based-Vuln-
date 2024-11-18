static void niagara_init(MachineState *machine)

{

    NiagaraBoardState *s = g_new(NiagaraBoardState, 1);

    DriveInfo *dinfo = drive_get_next(IF_PFLASH);

    MemoryRegion *sysmem = get_system_memory();



    /* init CPUs */

    sparc64_cpu_devinit(machine->cpu_model, "Sun UltraSparc T1",

                        NIAGARA_PROM_BASE);

    /* set up devices */

    memory_region_allocate_system_memory(&s->hv_ram, NULL, "sun4v-hv.ram",

                                         NIAGARA_HV_RAM_SIZE);

    memory_region_add_subregion(sysmem, NIAGARA_HV_RAM_BASE, &s->hv_ram);



    memory_region_allocate_system_memory(&s->partition_ram, NULL,

                                         "sun4v-partition.ram",

                                         machine->ram_size);

    memory_region_add_subregion(sysmem, NIAGARA_PARTITION_RAM_BASE,

                                &s->partition_ram);



    memory_region_allocate_system_memory(&s->nvram, NULL,

                                         "sun4v.nvram", NIAGARA_NVRAM_SIZE);

    memory_region_add_subregion(sysmem, NIAGARA_NVRAM_BASE, &s->nvram);

    memory_region_allocate_system_memory(&s->md_rom, NULL,

                                         "sun4v-md.rom", NIAGARA_MD_ROM_SIZE);

    memory_region_add_subregion(sysmem, NIAGARA_MD_ROM_BASE, &s->md_rom);

    memory_region_allocate_system_memory(&s->hv_rom, NULL,

                                         "sun4v-hv.rom", NIAGARA_HV_ROM_SIZE);

    memory_region_add_subregion(sysmem, NIAGARA_HV_ROM_BASE, &s->hv_rom);

    memory_region_allocate_system_memory(&s->prom, NULL,

                                         "sun4v.prom", PROM_SIZE_MAX);

    memory_region_add_subregion(sysmem, NIAGARA_PROM_BASE, &s->prom);



    add_rom_or_fail("nvram1", NIAGARA_NVRAM_BASE);

    add_rom_or_fail("1up-md.bin", NIAGARA_MD_ROM_BASE);

    add_rom_or_fail("1up-hv.bin", NIAGARA_HV_ROM_BASE);



    add_rom_or_fail("reset.bin", NIAGARA_PROM_BASE);

    add_rom_or_fail("q.bin", NIAGARA_PROM_BASE + NIAGARA_Q_OFFSET);

    add_rom_or_fail("openboot.bin", NIAGARA_PROM_BASE + NIAGARA_OBP_OFFSET);



    /* the virtual ramdisk is kind of initrd, but it resides

       outside of the partition RAM */

    if (dinfo) {

        BlockBackend *blk = blk_by_legacy_dinfo(dinfo);

        int size = blk_getlength(blk);

        if (size > 0) {

            memory_region_allocate_system_memory(&s->vdisk_ram, NULL,

                                                 "sun4v_vdisk.ram", size);

            memory_region_add_subregion(get_system_memory(),

                                        NIAGARA_VDISK_BASE, &s->vdisk_ram);

            dinfo->is_default = 1;

            rom_add_file_fixed(blk_bs(blk)->filename, NIAGARA_VDISK_BASE, -1);

        } else {

            fprintf(stderr, "qemu: could not load ram disk '%s'\n",

                    blk_bs(blk)->filename);

            exit(1);

        }

    }

    serial_mm_init(sysmem, NIAGARA_UART_BASE, 0, NULL, 115200,

                   serial_hds[0], DEVICE_BIG_ENDIAN);



    empty_slot_init(NIAGARA_IOBBASE, NIAGARA_IOBSIZE);

    sun4v_rtc_init(NIAGARA_RTC_BASE);

}

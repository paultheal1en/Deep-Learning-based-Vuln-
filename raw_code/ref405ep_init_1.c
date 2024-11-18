static void ref405ep_init(MachineState *machine)

{

    ram_addr_t ram_size = machine->ram_size;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    const char *initrd_filename = machine->initrd_filename;

    char *filename;

    ppc4xx_bd_info_t bd;

    CPUPPCState *env;

    qemu_irq *pic;

    MemoryRegion *bios;

    MemoryRegion *sram = g_new(MemoryRegion, 1);

    ram_addr_t bdloc;

    MemoryRegion *ram_memories = g_malloc(2 * sizeof(*ram_memories));

    hwaddr ram_bases[2], ram_sizes[2];

    target_ulong sram_size;

    long bios_size;

    //int phy_addr = 0;

    //static int phy_addr = 1;

    target_ulong kernel_base, initrd_base;

    long kernel_size, initrd_size;

    int linux_boot;

    int fl_idx, fl_sectors, len;

    DriveInfo *dinfo;

    MemoryRegion *sysmem = get_system_memory();



    /* XXX: fix this */

    memory_region_allocate_system_memory(&ram_memories[0], NULL, "ef405ep.ram",

                                         0x08000000);

    ram_bases[0] = 0;

    ram_sizes[0] = 0x08000000;

    memory_region_init(&ram_memories[1], NULL, "ef405ep.ram1", 0);

    ram_bases[1] = 0x00000000;

    ram_sizes[1] = 0x00000000;

    ram_size = 128 * 1024 * 1024;

#ifdef DEBUG_BOARD_INIT

    printf("%s: register cpu\n", __func__);

#endif

    env = ppc405ep_init(sysmem, ram_memories, ram_bases, ram_sizes,

                        33333333, &pic, kernel_filename == NULL ? 0 : 1);

    /* allocate SRAM */

    sram_size = 512 * 1024;

    memory_region_init_ram(sram, NULL, "ef405ep.sram", sram_size, &error_abort);

    vmstate_register_ram_global(sram);

    memory_region_add_subregion(sysmem, 0xFFF00000, sram);

    /* allocate and load BIOS */

#ifdef DEBUG_BOARD_INIT

    printf("%s: register BIOS\n", __func__);

#endif

    fl_idx = 0;

#ifdef USE_FLASH_BIOS

    dinfo = drive_get(IF_PFLASH, 0, fl_idx);

    if (dinfo) {

        BlockBackend *blk = blk_by_legacy_dinfo(dinfo);



        bios_size = blk_getlength(blk);

        fl_sectors = (bios_size + 65535) >> 16;

#ifdef DEBUG_BOARD_INIT

        printf("Register parallel flash %d size %lx"

               " at addr %lx '%s' %d\n",

               fl_idx, bios_size, -bios_size,

               blk_name(blk), fl_sectors);

#endif

        pflash_cfi02_register((uint32_t)(-bios_size),

                              NULL, "ef405ep.bios", bios_size,

                              blk, 65536, fl_sectors, 1,

                              2, 0x0001, 0x22DA, 0x0000, 0x0000, 0x555, 0x2AA,

                              1);

        fl_idx++;

    } else

#endif

    {

#ifdef DEBUG_BOARD_INIT

        printf("Load BIOS from file\n");

#endif

        bios = g_new(MemoryRegion, 1);

        memory_region_init_ram(bios, NULL, "ef405ep.bios", BIOS_SIZE,

                               &error_abort);

        vmstate_register_ram_global(bios);



        if (bios_name == NULL)

            bios_name = BIOS_FILENAME;

        filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

        if (filename) {

            bios_size = load_image(filename, memory_region_get_ram_ptr(bios));

            g_free(filename);

            if (bios_size < 0 || bios_size > BIOS_SIZE) {

                error_report("Could not load PowerPC BIOS '%s'", bios_name);

                exit(1);

            }

            bios_size = (bios_size + 0xfff) & ~0xfff;

            memory_region_add_subregion(sysmem, (uint32_t)(-bios_size), bios);

        } else if (!qtest_enabled() || kernel_filename != NULL) {

            error_report("Could not load PowerPC BIOS '%s'", bios_name);

            exit(1);

        } else {

            /* Avoid an uninitialized variable warning */

            bios_size = -1;

        }

        memory_region_set_readonly(bios, true);

    }

    /* Register FPGA */

#ifdef DEBUG_BOARD_INIT

    printf("%s: register FPGA\n", __func__);

#endif

    ref405ep_fpga_init(sysmem, 0xF0300000);

    /* Register NVRAM */

#ifdef DEBUG_BOARD_INIT

    printf("%s: register NVRAM\n", __func__);

#endif

    m48t59_init(NULL, 0xF0000000, 0, 8192, 1968, 8);

    /* Load kernel */

    linux_boot = (kernel_filename != NULL);

    if (linux_boot) {

#ifdef DEBUG_BOARD_INIT

        printf("%s: load kernel\n", __func__);

#endif

        memset(&bd, 0, sizeof(bd));

        bd.bi_memstart = 0x00000000;

        bd.bi_memsize = ram_size;

        bd.bi_flashstart = -bios_size;

        bd.bi_flashsize = -bios_size;

        bd.bi_flashoffset = 0;

        bd.bi_sramstart = 0xFFF00000;

        bd.bi_sramsize = sram_size;

        bd.bi_bootflags = 0;

        bd.bi_intfreq = 133333333;

        bd.bi_busfreq = 33333333;

        bd.bi_baudrate = 115200;

        bd.bi_s_version[0] = 'Q';

        bd.bi_s_version[1] = 'M';

        bd.bi_s_version[2] = 'U';

        bd.bi_s_version[3] = '\0';

        bd.bi_r_version[0] = 'Q';

        bd.bi_r_version[1] = 'E';

        bd.bi_r_version[2] = 'M';

        bd.bi_r_version[3] = 'U';

        bd.bi_r_version[4] = '\0';

        bd.bi_procfreq = 133333333;

        bd.bi_plb_busfreq = 33333333;

        bd.bi_pci_busfreq = 33333333;

        bd.bi_opbfreq = 33333333;

        bdloc = ppc405_set_bootinfo(env, &bd, 0x00000001);

        env->gpr[3] = bdloc;

        kernel_base = KERNEL_LOAD_ADDR;

        /* now we can load the kernel */

        kernel_size = load_image_targphys(kernel_filename, kernel_base,

                                          ram_size - kernel_base);

        if (kernel_size < 0) {

            fprintf(stderr, "qemu: could not load kernel '%s'\n",

                    kernel_filename);

            exit(1);

        }

        printf("Load kernel size %ld at " TARGET_FMT_lx,

               kernel_size, kernel_base);

        /* load initrd */

        if (initrd_filename) {

            initrd_base = INITRD_LOAD_ADDR;

            initrd_size = load_image_targphys(initrd_filename, initrd_base,

                                              ram_size - initrd_base);

            if (initrd_size < 0) {

                fprintf(stderr, "qemu: could not load initial ram disk '%s'\n",

                        initrd_filename);

                exit(1);

            }

        } else {

            initrd_base = 0;

            initrd_size = 0;

        }

        env->gpr[4] = initrd_base;

        env->gpr[5] = initrd_size;

        if (kernel_cmdline != NULL) {

            len = strlen(kernel_cmdline);

            bdloc -= ((len + 255) & ~255);

            cpu_physical_memory_write(bdloc, kernel_cmdline, len + 1);

            env->gpr[6] = bdloc;

            env->gpr[7] = bdloc + len;

        } else {

            env->gpr[6] = 0;

            env->gpr[7] = 0;

        }

        env->nip = KERNEL_LOAD_ADDR;

    } else {

        kernel_base = 0;

        kernel_size = 0;

        initrd_base = 0;

        initrd_size = 0;

        bdloc = 0;

    }

#ifdef DEBUG_BOARD_INIT

    printf("bdloc " RAM_ADDR_FMT "\n", bdloc);

    printf("%s: Done\n", __func__);

#endif

}

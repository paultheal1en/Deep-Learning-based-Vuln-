static void mainstone_common_init(ram_addr_t ram_size,

                const char *kernel_filename,

                const char *kernel_cmdline, const char *initrd_filename,

                const char *cpu_model, enum mainstone_model_e model, int arm_id)

{

    uint32_t sector_len = 256 * 1024;

    target_phys_addr_t mainstone_flash_base[] = { MST_FLASH_0, MST_FLASH_1 };

    PXA2xxState *cpu;

    DeviceState *mst_irq;

    DriveInfo *dinfo;

    int i;

    int be;



    if (!cpu_model)

        cpu_model = "pxa270-c5";



    /* Setup CPU & memory */

    cpu = pxa270_init(mainstone_binfo.ram_size, cpu_model);

    cpu_register_physical_memory(0, MAINSTONE_ROM,

                    qemu_ram_alloc(NULL, "mainstone.rom",

                                   MAINSTONE_ROM) | IO_MEM_ROM);



#ifdef TARGET_WORDS_BIGENDIAN

    be = 1;

#else

    be = 0;

#endif

    /* There are two 32MiB flash devices on the board */

    for (i = 0; i < 2; i ++) {

        dinfo = drive_get(IF_PFLASH, 0, i);

        if (!dinfo) {

            fprintf(stderr, "Two flash images must be given with the "

                    "'pflash' parameter\n");

            exit(1);

        }



        if (!pflash_cfi01_register(mainstone_flash_base[i],

                                   qemu_ram_alloc(NULL, i ? "mainstone.flash1" :

                                                  "mainstone.flash0",

                                                  MAINSTONE_FLASH),

                                   dinfo->bdrv, sector_len,

                                   MAINSTONE_FLASH / sector_len, 4, 0, 0, 0, 0,

                                   be)) {

            fprintf(stderr, "qemu: Error registering flash memory.\n");

            exit(1);

        }

    }



    mst_irq = sysbus_create_simple("mainstone-fpga", MST_FPGA_PHYS,

                    cpu->pic[PXA2XX_PIC_GPIO_0]);



    /* setup keypad */

    printf("map addr %p\n", &map);

    pxa27x_register_keypad(cpu->kp, map, 0xe0);



    /* MMC/SD host */

    pxa2xx_mmci_handlers(cpu->mmc, NULL, qdev_get_gpio_in(mst_irq, MMC_IRQ));



    smc91c111_init(&nd_table[0], MST_ETH_PHYS,

                    qdev_get_gpio_in(mst_irq, ETHERNET_IRQ));



    mainstone_binfo.kernel_filename = kernel_filename;

    mainstone_binfo.kernel_cmdline = kernel_cmdline;

    mainstone_binfo.initrd_filename = initrd_filename;

    mainstone_binfo.board_id = arm_id;

    arm_load_kernel(cpu->env, &mainstone_binfo);

}

static void vexpress_common_init(MachineState *machine)

{

    VexpressMachineState *vms = VEXPRESS_MACHINE(machine);

    VexpressMachineClass *vmc = VEXPRESS_MACHINE_GET_CLASS(machine);

    VEDBoardInfo *daughterboard = vmc->daughterboard;

    DeviceState *dev, *sysctl, *pl041;

    qemu_irq pic[64];

    uint32_t sys_id;

    DriveInfo *dinfo;

    pflash_t *pflash0;

    ram_addr_t vram_size, sram_size;

    MemoryRegion *sysmem = get_system_memory();

    MemoryRegion *vram = g_new(MemoryRegion, 1);

    MemoryRegion *sram = g_new(MemoryRegion, 1);

    MemoryRegion *flashalias = g_new(MemoryRegion, 1);

    MemoryRegion *flash0mem;

    const hwaddr *map = daughterboard->motherboard_map;

    int i;



    daughterboard->init(vms, machine->ram_size, machine->cpu_model, pic);



    /*

     * If a bios file was provided, attempt to map it into memory

     */

    if (bios_name) {

        char *fn;

        int image_size;



        if (drive_get(IF_PFLASH, 0, 0)) {

            error_report("The contents of the first flash device may be "

                         "specified with -bios or with -drive if=pflash... "

                         "but you cannot use both options at once");

            exit(1);

        }

        fn = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

        if (!fn) {

            error_report("Could not find ROM image '%s'", bios_name);

            exit(1);

        }

        image_size = load_image_targphys(fn, map[VE_NORFLASH0],

                                         VEXPRESS_FLASH_SIZE);

        g_free(fn);

        if (image_size < 0) {

            error_report("Could not load ROM image '%s'", bios_name);

            exit(1);

        }

    }



    /* Motherboard peripherals: the wiring is the same but the

     * addresses vary between the legacy and A-Series memory maps.

     */



    sys_id = 0x1190f500;



    sysctl = qdev_create(NULL, "realview_sysctl");

    qdev_prop_set_uint32(sysctl, "sys_id", sys_id);

    qdev_prop_set_uint32(sysctl, "proc_id", daughterboard->proc_id);

    qdev_prop_set_uint32(sysctl, "len-db-voltage",

                         daughterboard->num_voltage_sensors);

    for (i = 0; i < daughterboard->num_voltage_sensors; i++) {

        char *propname = g_strdup_printf("db-voltage[%d]", i);

        qdev_prop_set_uint32(sysctl, propname, daughterboard->voltages[i]);

        g_free(propname);

    }

    qdev_prop_set_uint32(sysctl, "len-db-clock",

                         daughterboard->num_clocks);

    for (i = 0; i < daughterboard->num_clocks; i++) {

        char *propname = g_strdup_printf("db-clock[%d]", i);

        qdev_prop_set_uint32(sysctl, propname, daughterboard->clocks[i]);

        g_free(propname);

    }

    qdev_init_nofail(sysctl);

    sysbus_mmio_map(SYS_BUS_DEVICE(sysctl), 0, map[VE_SYSREGS]);



    /* VE_SP810: not modelled */

    /* VE_SERIALPCI: not modelled */



    pl041 = qdev_create(NULL, "pl041");

    qdev_prop_set_uint32(pl041, "nc_fifo_depth", 512);

    qdev_init_nofail(pl041);

    sysbus_mmio_map(SYS_BUS_DEVICE(pl041), 0, map[VE_PL041]);

    sysbus_connect_irq(SYS_BUS_DEVICE(pl041), 0, pic[11]);



    dev = sysbus_create_varargs("pl181", map[VE_MMCI], pic[9], pic[10], NULL);

    /* Wire up MMC card detect and read-only signals */

    qdev_connect_gpio_out(dev, 0,

                          qdev_get_gpio_in(sysctl, ARM_SYSCTL_GPIO_MMC_WPROT));

    qdev_connect_gpio_out(dev, 1,

                          qdev_get_gpio_in(sysctl, ARM_SYSCTL_GPIO_MMC_CARDIN));



    sysbus_create_simple("pl050_keyboard", map[VE_KMI0], pic[12]);

    sysbus_create_simple("pl050_mouse", map[VE_KMI1], pic[13]);



    sysbus_create_simple("pl011", map[VE_UART0], pic[5]);

    sysbus_create_simple("pl011", map[VE_UART1], pic[6]);

    sysbus_create_simple("pl011", map[VE_UART2], pic[7]);

    sysbus_create_simple("pl011", map[VE_UART3], pic[8]);



    sysbus_create_simple("sp804", map[VE_TIMER01], pic[2]);

    sysbus_create_simple("sp804", map[VE_TIMER23], pic[3]);



    /* VE_SERIALDVI: not modelled */



    sysbus_create_simple("pl031", map[VE_RTC], pic[4]); /* RTC */



    /* VE_COMPACTFLASH: not modelled */



    sysbus_create_simple("pl111", map[VE_CLCD], pic[14]);



    dinfo = drive_get_next(IF_PFLASH);

    pflash0 = ve_pflash_cfi01_register(map[VE_NORFLASH0], "vexpress.flash0",

                                       dinfo);

    if (!pflash0) {

        fprintf(stderr, "vexpress: error registering flash 0.\n");

        exit(1);

    }



    if (map[VE_NORFLASHALIAS] != -1) {

        /* Map flash 0 as an alias into low memory */

        flash0mem = sysbus_mmio_get_region(SYS_BUS_DEVICE(pflash0), 0);

        memory_region_init_alias(flashalias, NULL, "vexpress.flashalias",

                                 flash0mem, 0, VEXPRESS_FLASH_SIZE);

        memory_region_add_subregion(sysmem, map[VE_NORFLASHALIAS], flashalias);

    }



    dinfo = drive_get_next(IF_PFLASH);

    if (!ve_pflash_cfi01_register(map[VE_NORFLASH1], "vexpress.flash1",

                                  dinfo)) {

        fprintf(stderr, "vexpress: error registering flash 1.\n");

        exit(1);

    }



    sram_size = 0x2000000;

    memory_region_init_ram(sram, NULL, "vexpress.sram", sram_size,

                           &error_abort);

    vmstate_register_ram_global(sram);

    memory_region_add_subregion(sysmem, map[VE_SRAM], sram);



    vram_size = 0x800000;

    memory_region_init_ram(vram, NULL, "vexpress.vram", vram_size,

                           &error_abort);

    vmstate_register_ram_global(vram);

    memory_region_add_subregion(sysmem, map[VE_VIDEORAM], vram);



    /* 0x4e000000 LAN9118 Ethernet */

    if (nd_table[0].used) {

        lan9118_init(&nd_table[0], map[VE_ETHERNET], pic[15]);

    }



    /* VE_USB: not modelled */



    /* VE_DAPROM: not modelled */



    /* Create mmio transports, so the user can create virtio backends

     * (which will be automatically plugged in to the transports). If

     * no backend is created the transport will just sit harmlessly idle.

     */

    for (i = 0; i < NUM_VIRTIO_TRANSPORTS; i++) {

        sysbus_create_simple("virtio-mmio", map[VE_VIRTIO] + 0x200 * i,

                             pic[40 + i]);

    }



    daughterboard->bootinfo.ram_size = machine->ram_size;

    daughterboard->bootinfo.kernel_filename = machine->kernel_filename;

    daughterboard->bootinfo.kernel_cmdline = machine->kernel_cmdline;

    daughterboard->bootinfo.initrd_filename = machine->initrd_filename;

    daughterboard->bootinfo.nb_cpus = smp_cpus;

    daughterboard->bootinfo.board_id = VEXPRESS_BOARD_ID;

    daughterboard->bootinfo.loader_start = daughterboard->loader_start;

    daughterboard->bootinfo.smp_loader_start = map[VE_SRAM];

    daughterboard->bootinfo.smp_bootreg_addr = map[VE_SYSREGS] + 0x30;

    daughterboard->bootinfo.gic_cpu_if_addr = daughterboard->gic_cpu_if_addr;

    daughterboard->bootinfo.modify_dtb = vexpress_modify_dtb;

    /* Indicate that when booting Linux we should be in secure state */

    daughterboard->bootinfo.secure_boot = true;

    arm_load_kernel(ARM_CPU(first_cpu), &daughterboard->bootinfo);

}

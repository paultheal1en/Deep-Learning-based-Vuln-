static void versatile_init(QEMUMachineInitArgs *args, int board_id)

{

    ARMCPU *cpu;

    MemoryRegion *sysmem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    qemu_irq *cpu_pic;

    qemu_irq pic[32];

    qemu_irq sic[32];

    DeviceState *dev, *sysctl;

    SysBusDevice *busdev;

    DeviceState *pl041;

    PCIBus *pci_bus;

    NICInfo *nd;

    i2c_bus *i2c;

    int n;

    int done_smc = 0;

    DriveInfo *dinfo;



    if (!args->cpu_model) {

        args->cpu_model = "arm926";

    }

    cpu = cpu_arm_init(args->cpu_model);

    if (!cpu) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    memory_region_init_ram(ram, "versatile.ram", args->ram_size);

    vmstate_register_ram_global(ram);

    /* ??? RAM should repeat to fill physical memory space.  */

    /* SDRAM at address zero.  */

    memory_region_add_subregion(sysmem, 0, ram);



    sysctl = qdev_create(NULL, "realview_sysctl");

    qdev_prop_set_uint32(sysctl, "sys_id", 0x41007004);

    qdev_prop_set_uint32(sysctl, "proc_id", 0x02000000);

    qdev_init_nofail(sysctl);

    sysbus_mmio_map(SYS_BUS_DEVICE(sysctl), 0, 0x10000000);



    cpu_pic = arm_pic_init_cpu(cpu);

    dev = sysbus_create_varargs("pl190", 0x10140000,

                                cpu_pic[ARM_PIC_CPU_IRQ],

                                cpu_pic[ARM_PIC_CPU_FIQ], NULL);

    for (n = 0; n < 32; n++) {

        pic[n] = qdev_get_gpio_in(dev, n);

    }

    dev = sysbus_create_simple("versatilepb_sic", 0x10003000, NULL);

    for (n = 0; n < 32; n++) {

        sysbus_connect_irq(SYS_BUS_DEVICE(dev), n, pic[n]);

        sic[n] = qdev_get_gpio_in(dev, n);

    }



    sysbus_create_simple("pl050_keyboard", 0x10006000, sic[3]);

    sysbus_create_simple("pl050_mouse", 0x10007000, sic[4]);



    dev = qdev_create(NULL, "versatile_pci");

    busdev = SYS_BUS_DEVICE(dev);

    qdev_init_nofail(dev);

    sysbus_mmio_map(busdev, 0, 0x41000000); /* PCI self-config */

    sysbus_mmio_map(busdev, 1, 0x42000000); /* PCI config */

    sysbus_connect_irq(busdev, 0, sic[27]);

    sysbus_connect_irq(busdev, 1, sic[28]);

    sysbus_connect_irq(busdev, 2, sic[29]);

    sysbus_connect_irq(busdev, 3, sic[30]);

    pci_bus = (PCIBus *)qdev_get_child_bus(dev, "pci");



    /* The Versatile PCI bridge does not provide access to PCI IO space,

       so many of the qemu PCI devices are not useable.  */

    for(n = 0; n < nb_nics; n++) {

        nd = &nd_table[n];



        if (!done_smc && (!nd->model || strcmp(nd->model, "smc91c111") == 0)) {

            smc91c111_init(nd, 0x10010000, sic[25]);

            done_smc = 1;

        } else {

            pci_nic_init_nofail(nd, "rtl8139", NULL);

        }

    }

    if (usb_enabled(false)) {

        pci_create_simple(pci_bus, -1, "pci-ohci");

    }

    n = drive_get_max_bus(IF_SCSI);

    while (n >= 0) {

        pci_create_simple(pci_bus, -1, "lsi53c895a");

        n--;

    }



    sysbus_create_simple("pl011", 0x101f1000, pic[12]);

    sysbus_create_simple("pl011", 0x101f2000, pic[13]);

    sysbus_create_simple("pl011", 0x101f3000, pic[14]);

    sysbus_create_simple("pl011", 0x10009000, sic[6]);



    sysbus_create_simple("pl080", 0x10130000, pic[17]);

    sysbus_create_simple("sp804", 0x101e2000, pic[4]);

    sysbus_create_simple("sp804", 0x101e3000, pic[5]);



    sysbus_create_simple("pl061", 0x101e4000, pic[6]);

    sysbus_create_simple("pl061", 0x101e5000, pic[7]);

    sysbus_create_simple("pl061", 0x101e6000, pic[8]);

    sysbus_create_simple("pl061", 0x101e7000, pic[9]);



    /* The versatile/PB actually has a modified Color LCD controller

       that includes hardware cursor support from the PL111.  */

    dev = sysbus_create_simple("pl110_versatile", 0x10120000, pic[16]);

    /* Wire up the mux control signals from the SYS_CLCD register */

    qdev_connect_gpio_out(sysctl, 0, qdev_get_gpio_in(dev, 0));



    sysbus_create_varargs("pl181", 0x10005000, sic[22], sic[1], NULL);

    sysbus_create_varargs("pl181", 0x1000b000, sic[23], sic[2], NULL);



    /* Add PL031 Real Time Clock. */

    sysbus_create_simple("pl031", 0x101e8000, pic[10]);



    dev = sysbus_create_simple("versatile_i2c", 0x10002000, NULL);

    i2c = (i2c_bus *)qdev_get_child_bus(dev, "i2c");

    i2c_create_slave(i2c, "ds1338", 0x68);



    /* Add PL041 AACI Interface to the LM4549 codec */

    pl041 = qdev_create(NULL, "pl041");

    qdev_prop_set_uint32(pl041, "nc_fifo_depth", 512);

    qdev_init_nofail(pl041);

    sysbus_mmio_map(SYS_BUS_DEVICE(pl041), 0, 0x10004000);

    sysbus_connect_irq(SYS_BUS_DEVICE(pl041), 0, sic[24]);



    /* Memory map for Versatile/PB:  */

    /* 0x10000000 System registers.  */

    /* 0x10001000 PCI controller config registers.  */

    /* 0x10002000 Serial bus interface.  */

    /*  0x10003000 Secondary interrupt controller.  */

    /* 0x10004000 AACI (audio).  */

    /*  0x10005000 MMCI0.  */

    /*  0x10006000 KMI0 (keyboard).  */

    /*  0x10007000 KMI1 (mouse).  */

    /* 0x10008000 Character LCD Interface.  */

    /*  0x10009000 UART3.  */

    /* 0x1000a000 Smart card 1.  */

    /*  0x1000b000 MMCI1.  */

    /*  0x10010000 Ethernet.  */

    /* 0x10020000 USB.  */

    /* 0x10100000 SSMC.  */

    /* 0x10110000 MPMC.  */

    /*  0x10120000 CLCD Controller.  */

    /*  0x10130000 DMA Controller.  */

    /*  0x10140000 Vectored interrupt controller.  */

    /* 0x101d0000 AHB Monitor Interface.  */

    /* 0x101e0000 System Controller.  */

    /* 0x101e1000 Watchdog Interface.  */

    /* 0x101e2000 Timer 0/1.  */

    /* 0x101e3000 Timer 2/3.  */

    /* 0x101e4000 GPIO port 0.  */

    /* 0x101e5000 GPIO port 1.  */

    /* 0x101e6000 GPIO port 2.  */

    /* 0x101e7000 GPIO port 3.  */

    /* 0x101e8000 RTC.  */

    /* 0x101f0000 Smart card 0.  */

    /*  0x101f1000 UART0.  */

    /*  0x101f2000 UART1.  */

    /*  0x101f3000 UART2.  */

    /* 0x101f4000 SSPI.  */

    /* 0x34000000 NOR Flash */



    dinfo = drive_get(IF_PFLASH, 0, 0);

    if (!pflash_cfi01_register(VERSATILE_FLASH_ADDR, NULL, "versatile.flash",

                          VERSATILE_FLASH_SIZE, dinfo ? dinfo->bdrv : NULL,

                          VERSATILE_FLASH_SECT_SIZE,

                          VERSATILE_FLASH_SIZE / VERSATILE_FLASH_SECT_SIZE,

                          4, 0x0089, 0x0018, 0x0000, 0x0, 0)) {

        fprintf(stderr, "qemu: Error registering flash memory.\n");

    }



    versatile_binfo.ram_size = args->ram_size;

    versatile_binfo.kernel_filename = args->kernel_filename;

    versatile_binfo.kernel_cmdline = args->kernel_cmdline;

    versatile_binfo.initrd_filename = args->initrd_filename;

    versatile_binfo.board_id = board_id;

    arm_load_kernel(cpu, &versatile_binfo);

}

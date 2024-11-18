static void versatile_init(ram_addr_t ram_size,

                     const char *boot_device,

                     const char *kernel_filename, const char *kernel_cmdline,

                     const char *initrd_filename, const char *cpu_model,

                     int board_id)

{

    CPUState *env;

    ram_addr_t ram_offset;

    qemu_irq *cpu_pic;

    qemu_irq pic[32];

    qemu_irq sic[32];

    DeviceState *dev;

    PCIBus *pci_bus;

    NICInfo *nd;

    int n;

    int done_smc = 0;



    if (!cpu_model)

        cpu_model = "arm926";

    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    ram_offset = qemu_ram_alloc(NULL, "versatile.ram", ram_size);

    /* ??? RAM should repeat to fill physical memory space.  */

    /* SDRAM at address zero.  */

    cpu_register_physical_memory(0, ram_size, ram_offset | IO_MEM_RAM);



    arm_sysctl_init(0x10000000, 0x41007004, 0x02000000);

    cpu_pic = arm_pic_init_cpu(env);

    dev = sysbus_create_varargs("pl190", 0x10140000,

                                cpu_pic[0], cpu_pic[1], NULL);

    for (n = 0; n < 32; n++) {

        pic[n] = qdev_get_gpio_in(dev, n);

    }

    dev = sysbus_create_simple("versatilepb_sic", 0x10003000, NULL);

    for (n = 0; n < 32; n++) {

        sysbus_connect_irq(sysbus_from_qdev(dev), n, pic[n]);

        sic[n] = qdev_get_gpio_in(dev, n);

    }



    sysbus_create_simple("pl050_keyboard", 0x10006000, sic[3]);

    sysbus_create_simple("pl050_mouse", 0x10007000, sic[4]);



    dev = sysbus_create_varargs("versatile_pci", 0x40000000,

                                sic[27], sic[28], sic[29], sic[30], NULL);

    pci_bus = (PCIBus *)qdev_get_child_bus(dev, "pci");



    /* The Versatile PCI bridge does not provide access to PCI IO space,

       so many of the qemu PCI devices are not useable.  */

    for(n = 0; n < nb_nics; n++) {

        nd = &nd_table[n];



        if ((!nd->model && !done_smc) || strcmp(nd->model, "smc91c111") == 0) {

            smc91c111_init(nd, 0x10010000, sic[25]);

            done_smc = 1;

        } else {

            pci_nic_init_nofail(nd, "rtl8139", NULL);

        }

    }

    if (usb_enabled) {

        usb_ohci_init_pci(pci_bus, -1);

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



    /* The versatile/PB actually has a modified Color LCD controller

       that includes hardware cursor support from the PL111.  */

    sysbus_create_simple("pl110_versatile", 0x10120000, pic[16]);



    sysbus_create_varargs("pl181", 0x10005000, sic[22], sic[1], NULL);

    sysbus_create_varargs("pl181", 0x1000b000, sic[23], sic[2], NULL);



    /* Add PL031 Real Time Clock. */

    sysbus_create_simple("pl031", 0x101e8000, pic[10]);



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



    versatile_binfo.ram_size = ram_size;

    versatile_binfo.kernel_filename = kernel_filename;

    versatile_binfo.kernel_cmdline = kernel_cmdline;

    versatile_binfo.initrd_filename = initrd_filename;

    versatile_binfo.board_id = board_id;

    arm_load_kernel(env, &versatile_binfo);

}

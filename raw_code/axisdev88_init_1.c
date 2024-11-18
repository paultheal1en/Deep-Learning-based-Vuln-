void axisdev88_init(MachineState *machine)

{

    ram_addr_t ram_size = machine->ram_size;

    const char *cpu_model = machine->cpu_model;

    const char *kernel_filename = machine->kernel_filename;

    const char *kernel_cmdline = machine->kernel_cmdline;

    CRISCPU *cpu;

    CPUCRISState *env;

    DeviceState *dev;

    SysBusDevice *s;

    DriveInfo *nand;

    qemu_irq irq[30], nmi[2];

    void *etraxfs_dmac;

    struct etraxfs_dma_client *dma_eth;

    int i;

    MemoryRegion *address_space_mem = get_system_memory();

    MemoryRegion *phys_ram = g_new(MemoryRegion, 1);

    MemoryRegion *phys_intmem = g_new(MemoryRegion, 1);



    /* init CPUs */

    if (cpu_model == NULL) {

        cpu_model = "crisv32";

    }

    cpu = cpu_cris_init(cpu_model);

    env = &cpu->env;



    /* allocate RAM */

    memory_region_allocate_system_memory(phys_ram, NULL, "axisdev88.ram",

                                         ram_size);

    memory_region_add_subregion(address_space_mem, 0x40000000, phys_ram);



    /* The ETRAX-FS has 128Kb on chip ram, the docs refer to it as the 

       internal memory.  */

    memory_region_init_ram(phys_intmem, NULL, "axisdev88.chipram", INTMEM_SIZE,

                           &error_abort);

    vmstate_register_ram_global(phys_intmem);

    memory_region_add_subregion(address_space_mem, 0x38000000, phys_intmem);



      /* Attach a NAND flash to CS1.  */

    nand = drive_get(IF_MTD, 0, 0);

    nand_state.nand = nand_init(nand ? blk_by_legacy_dinfo(nand) : NULL,

                                NAND_MFR_STMICRO, 0x39);

    memory_region_init_io(&nand_state.iomem, NULL, &nand_ops, &nand_state,

                          "nand", 0x05000000);

    memory_region_add_subregion(address_space_mem, 0x10000000,

                                &nand_state.iomem);



    gpio_state.nand = &nand_state;

    memory_region_init_io(&gpio_state.iomem, NULL, &gpio_ops, &gpio_state,

                          "gpio", 0x5c);

    memory_region_add_subregion(address_space_mem, 0x3001a000,

                                &gpio_state.iomem);





    dev = qdev_create(NULL, "etraxfs,pic");

    /* FIXME: Is there a proper way to signal vectors to the CPU core?  */

    qdev_prop_set_ptr(dev, "interrupt_vector", &env->interrupt_vector);

    qdev_init_nofail(dev);

    s = SYS_BUS_DEVICE(dev);

    sysbus_mmio_map(s, 0, 0x3001c000);

    sysbus_connect_irq(s, 0, qdev_get_gpio_in(DEVICE(cpu), CRIS_CPU_IRQ));

    sysbus_connect_irq(s, 1, qdev_get_gpio_in(DEVICE(cpu), CRIS_CPU_NMI));

    for (i = 0; i < 30; i++) {

        irq[i] = qdev_get_gpio_in(dev, i);

    }

    nmi[0] = qdev_get_gpio_in(dev, 30);

    nmi[1] = qdev_get_gpio_in(dev, 31);



    etraxfs_dmac = etraxfs_dmac_init(0x30000000, 10);

    for (i = 0; i < 10; i++) {

        /* On ETRAX, odd numbered channels are inputs.  */

        etraxfs_dmac_connect(etraxfs_dmac, i, irq + 7 + i, i & 1);

    }



    /* Add the two ethernet blocks.  */

    dma_eth = g_malloc0(sizeof dma_eth[0] * 4); /* Allocate 4 channels.  */

    etraxfs_eth_init(&nd_table[0], 0x30034000, 1, &dma_eth[0], &dma_eth[1]);

    if (nb_nics > 1) {

        etraxfs_eth_init(&nd_table[1], 0x30036000, 2, &dma_eth[2], &dma_eth[3]);

    }



    /* The DMA Connector block is missing, hardwire things for now.  */

    etraxfs_dmac_connect_client(etraxfs_dmac, 0, &dma_eth[0]);

    etraxfs_dmac_connect_client(etraxfs_dmac, 1, &dma_eth[1]);

    if (nb_nics > 1) {

        etraxfs_dmac_connect_client(etraxfs_dmac, 6, &dma_eth[2]);

        etraxfs_dmac_connect_client(etraxfs_dmac, 7, &dma_eth[3]);

    }



    /* 2 timers.  */

    sysbus_create_varargs("etraxfs,timer", 0x3001e000, irq[0x1b], nmi[1], NULL);

    sysbus_create_varargs("etraxfs,timer", 0x3005e000, irq[0x1b], nmi[1], NULL);



    for (i = 0; i < 4; i++) {

        sysbus_create_simple("etraxfs,serial", 0x30026000 + i * 0x2000,

                             irq[0x14 + i]);

    }



    if (kernel_filename) {

        li.image_filename = kernel_filename;

        li.cmdline = kernel_cmdline;

        cris_load_image(cpu, &li);

    } else if (!qtest_enabled()) {

        fprintf(stderr, "Kernel image must be specified\n");

        exit(1);

    }

}

static void ibm_40p_init(MachineState *machine)

{

    CPUPPCState *env = NULL;

    uint16_t cmos_checksum;

    PowerPCCPU *cpu;

    DeviceState *dev;

    SysBusDevice *pcihost;

    Nvram *m48t59 = NULL;

    PCIBus *pci_bus;

    ISABus *isa_bus;

    void *fw_cfg;

    int i;

    uint32_t kernel_base = 0, initrd_base = 0;

    long kernel_size = 0, initrd_size = 0;

    char boot_device;



    /* init CPU */

    if (!machine->cpu_model) {

        machine->cpu_model = "604";

    }

    cpu = POWERPC_CPU(cpu_generic_init(TYPE_POWERPC_CPU, machine->cpu_model));

    if (!cpu) {

        error_report("could not initialize CPU '%s'",

                     machine->cpu_model);

        exit(1);

    }

    env = &cpu->env;

    if (PPC_INPUT(env) != PPC_FLAGS_INPUT_6xx) {

        error_report("only 6xx bus is supported on this machine");

        exit(1);

    }



    if (env->flags & POWERPC_FLAG_RTC_CLK) {

        /* POWER / PowerPC 601 RTC clock frequency is 7.8125 MHz */

        cpu_ppc_tb_init(env, 7812500UL);

    } else {

        /* Set time-base frequency to 100 Mhz */

        cpu_ppc_tb_init(env, 100UL * 1000UL * 1000UL);

    }

    qemu_register_reset(ppc_prep_reset, cpu);



    /* PCI host */

    dev = qdev_create(NULL, "raven-pcihost");

    if (!bios_name) {

        bios_name = BIOS_FILENAME;

    }

    qdev_prop_set_string(dev, "bios-name", bios_name);

    qdev_prop_set_uint32(dev, "elf-machine", PPC_ELF_MACHINE);

    pcihost = SYS_BUS_DEVICE(dev);

    object_property_add_child(qdev_get_machine(), "raven", OBJECT(dev), NULL);

    qdev_init_nofail(dev);

    pci_bus = PCI_BUS(qdev_get_child_bus(dev, "pci.0"));

    if (!pci_bus) {

        error_report("could not create PCI host controller");

        exit(1);

    }



    /* PCI -> ISA bridge */

    dev = DEVICE(pci_create_simple(pci_bus, PCI_DEVFN(11, 0), "i82378"));

    qdev_connect_gpio_out(dev, 0,

                          cpu->env.irq_inputs[PPC6xx_INPUT_INT]);

    sysbus_connect_irq(pcihost, 0, qdev_get_gpio_in(dev, 15));

    sysbus_connect_irq(pcihost, 1, qdev_get_gpio_in(dev, 13));

    sysbus_connect_irq(pcihost, 2, qdev_get_gpio_in(dev, 15));

    sysbus_connect_irq(pcihost, 3, qdev_get_gpio_in(dev, 13));

    isa_bus = ISA_BUS(qdev_get_child_bus(dev, "isa.0"));



    /* Memory controller */

    dev = DEVICE(isa_create(isa_bus, "rs6000-mc"));

    qdev_prop_set_uint32(dev, "ram-size", machine->ram_size);

    qdev_init_nofail(dev);



    /* initialize CMOS checksums */

    cmos_checksum = 0x6aa9;

    qbus_walk_children(BUS(isa_bus), prep_set_cmos_checksum, NULL, NULL, NULL,

                       &cmos_checksum);



    /* add some more devices */

    if (defaults_enabled()) {

        isa_create_simple(isa_bus, "i8042");

        m48t59 = NVRAM(isa_create_simple(isa_bus, "isa-m48t59"));



        dev = DEVICE(isa_create(isa_bus, "cs4231a"));

        qdev_prop_set_uint32(dev, "iobase", 0x830);

        qdev_prop_set_uint32(dev, "irq", 10);

        qdev_init_nofail(dev);



        dev = DEVICE(isa_create(isa_bus, "pc87312"));

        qdev_prop_set_uint32(dev, "config", 12);

        qdev_init_nofail(dev);



        dev = DEVICE(isa_create(isa_bus, "prep-systemio"));

        qdev_prop_set_uint32(dev, "ibm-planar-id", 0xfc);

        qdev_prop_set_uint32(dev, "equipment", 0xc0);

        qdev_init_nofail(dev);



        pci_create_simple(pci_bus, PCI_DEVFN(1, 0), "lsi53c810");



        /* XXX: s3-trio at PCI_DEVFN(2, 0) */

        pci_vga_init(pci_bus);



        for (i = 0; i < nb_nics; i++) {

            pci_nic_init_nofail(&nd_table[i], pci_bus, "pcnet",

                                i == 0 ? "3" : NULL);

        }

    }



    /* Prepare firmware configuration for OpenBIOS */

    fw_cfg = fw_cfg_init_mem(CFG_ADDR, CFG_ADDR + 2);



    if (machine->kernel_filename) {

        /* load kernel */

        kernel_base = KERNEL_LOAD_ADDR;

        kernel_size = load_image_targphys(machine->kernel_filename,

                                          kernel_base,

                                          machine->ram_size - kernel_base);

        if (kernel_size < 0) {

            error_report("could not load kernel '%s'",

                         machine->kernel_filename);

            exit(1);

        }

        fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_ADDR, kernel_base);

        fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_SIZE, kernel_size);

        /* load initrd */

        if (machine->initrd_filename) {

            initrd_base = INITRD_LOAD_ADDR;

            initrd_size = load_image_targphys(machine->initrd_filename,

                                              initrd_base,

                                              machine->ram_size - initrd_base);

            if (initrd_size < 0) {

                error_report("could not load initial ram disk '%s'",

                             machine->initrd_filename);

                exit(1);

            }

            fw_cfg_add_i32(fw_cfg, FW_CFG_INITRD_ADDR, initrd_base);

            fw_cfg_add_i32(fw_cfg, FW_CFG_INITRD_SIZE, initrd_size);

        }

        if (machine->kernel_cmdline && *machine->kernel_cmdline) {

            fw_cfg_add_i32(fw_cfg, FW_CFG_KERNEL_CMDLINE, CMDLINE_ADDR);

            pstrcpy_targphys("cmdline", CMDLINE_ADDR, TARGET_PAGE_SIZE,

                             machine->kernel_cmdline);

            fw_cfg_add_string(fw_cfg, FW_CFG_CMDLINE_DATA,

                              machine->kernel_cmdline);

            fw_cfg_add_i32(fw_cfg, FW_CFG_CMDLINE_SIZE,

                           strlen(machine->kernel_cmdline) + 1);

        }

        boot_device = 'm';

    } else {

        boot_device = machine->boot_order[0];

    }



    fw_cfg_add_i16(fw_cfg, FW_CFG_MAX_CPUS, (uint16_t)max_cpus);

    fw_cfg_add_i64(fw_cfg, FW_CFG_RAM_SIZE, (uint64_t)machine->ram_size);

    fw_cfg_add_i16(fw_cfg, FW_CFG_MACHINE_ID, ARCH_PREP);



    fw_cfg_add_i16(fw_cfg, FW_CFG_PPC_WIDTH, graphic_width);

    fw_cfg_add_i16(fw_cfg, FW_CFG_PPC_HEIGHT, graphic_height);

    fw_cfg_add_i16(fw_cfg, FW_CFG_PPC_DEPTH, graphic_depth);



    fw_cfg_add_i32(fw_cfg, FW_CFG_PPC_IS_KVM, kvm_enabled());

    if (kvm_enabled()) {

#ifdef CONFIG_KVM

        uint8_t *hypercall;



        fw_cfg_add_i32(fw_cfg, FW_CFG_PPC_TBFREQ, kvmppc_get_tbfreq());

        hypercall = g_malloc(16);

        kvmppc_get_hypercall(env, hypercall, 16);

        fw_cfg_add_bytes(fw_cfg, FW_CFG_PPC_KVM_HC, hypercall, 16);

        fw_cfg_add_i32(fw_cfg, FW_CFG_PPC_KVM_PID, getpid());

#endif

    } else {

        fw_cfg_add_i32(fw_cfg, FW_CFG_PPC_TBFREQ, NANOSECONDS_PER_SECOND);

    }

    fw_cfg_add_i16(fw_cfg, FW_CFG_BOOT_DEVICE, boot_device);

    qemu_register_boot_set(fw_cfg_boot_set, fw_cfg);



    /* Prepare firmware configuration for Open Hack'Ware */

    if (m48t59) {

        PPC_NVRAM_set_params(m48t59, NVRAM_SIZE, "PREP", ram_size,

                             boot_device,

                             kernel_base, kernel_size,

                             machine->kernel_cmdline,

                             initrd_base, initrd_size,

                             /* XXX: need an option to load a NVRAM image */

                             0,

                             graphic_width, graphic_height, graphic_depth);

    }

}

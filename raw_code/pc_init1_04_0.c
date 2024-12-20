static void pc_init1(MemoryRegion *system_memory,

                     MemoryRegion *system_io,

                     ram_addr_t ram_size,

                     const char *boot_device,

                     const char *kernel_filename,

                     const char *kernel_cmdline,

                     const char *initrd_filename,

                     const char *cpu_model,

                     int pci_enabled,

                     int kvmclock_enabled)

{

    int i;

    ram_addr_t below_4g_mem_size, above_4g_mem_size;

    PCIBus *pci_bus;

    ISABus *isa_bus;

    PCII440FXState *i440fx_state;

    int piix3_devfn = -1;

    qemu_irq *cpu_irq;

    qemu_irq *gsi;

    qemu_irq *i8259;

    qemu_irq *smi_irq;

    GSIState *gsi_state;

    DriveInfo *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    BusState *idebus[MAX_IDE_BUS];

    ISADevice *rtc_state;

    ISADevice *floppy;

    MemoryRegion *ram_memory;

    MemoryRegion *pci_memory;

    MemoryRegion *rom_memory;

    DeviceState *icc_bridge;

    FWCfgState *fw_cfg = NULL;



    if (xen_enabled() && xen_hvm_init() != 0) {

        fprintf(stderr, "xen hardware virtual machine initialisation failed\n");

        exit(1);

    }



    icc_bridge = qdev_create(NULL, TYPE_ICC_BRIDGE);

    object_property_add_child(qdev_get_machine(), "icc-bridge",

                              OBJECT(icc_bridge), NULL);



    pc_cpus_init(cpu_model, icc_bridge);

    pc_acpi_init("acpi-dsdt.aml");



    if (kvm_enabled() && kvmclock_enabled) {

        kvmclock_create();

    }



    if (ram_size >= 0xe0000000 ) {

        above_4g_mem_size = ram_size - 0xe0000000;

        below_4g_mem_size = 0xe0000000;

    } else {

        above_4g_mem_size = 0;

        below_4g_mem_size = ram_size;

    }



    if (pci_enabled) {

        pci_memory = g_new(MemoryRegion, 1);

        memory_region_init(pci_memory, "pci", INT64_MAX);

        rom_memory = pci_memory;

    } else {

        pci_memory = NULL;

        rom_memory = system_memory;

    }



    /* allocate ram and load rom/bios */

    if (!xen_enabled()) {

        fw_cfg = pc_memory_init(system_memory,

                       kernel_filename, kernel_cmdline, initrd_filename,

                       below_4g_mem_size, above_4g_mem_size,

                       rom_memory, &ram_memory);

    }



    gsi_state = g_malloc0(sizeof(*gsi_state));

    if (kvm_irqchip_in_kernel()) {

        kvm_pc_setup_irq_routing(pci_enabled);

        gsi = qemu_allocate_irqs(kvm_pc_gsi_handler, gsi_state,

                                 GSI_NUM_PINS);

    } else {

        gsi = qemu_allocate_irqs(gsi_handler, gsi_state, GSI_NUM_PINS);

    }



    if (pci_enabled) {

        pci_bus = i440fx_init(&i440fx_state, &piix3_devfn, &isa_bus, gsi,

                              system_memory, system_io, ram_size,

                              below_4g_mem_size,

                              0x100000000ULL - below_4g_mem_size,

                              0x100000000ULL + above_4g_mem_size,

                              (sizeof(hwaddr) == 4

                               ? 0

                               : ((uint64_t)1 << 62)),

                              pci_memory, ram_memory);

    } else {

        pci_bus = NULL;

        i440fx_state = NULL;

        isa_bus = isa_bus_new(NULL, system_io);

        no_hpet = 1;

    }

    isa_bus_irqs(isa_bus, gsi);



    if (kvm_irqchip_in_kernel()) {

        i8259 = kvm_i8259_init(isa_bus);

    } else if (xen_enabled()) {

        i8259 = xen_interrupt_controller_init();

    } else {

        cpu_irq = pc_allocate_cpu_irq();

        i8259 = i8259_init(isa_bus, cpu_irq[0]);

    }



    for (i = 0; i < ISA_NUM_IRQS; i++) {

        gsi_state->i8259_irq[i] = i8259[i];

    }

    if (pci_enabled) {

        ioapic_init_gsi(gsi_state, "i440fx");

    }

    qdev_init_nofail(icc_bridge);



    pc_register_ferr_irq(gsi[13]);



    pc_vga_init(isa_bus, pci_enabled ? pci_bus : NULL);

    if (xen_enabled()) {

        pci_create_simple(pci_bus, -1, "xen-platform");

    }



    /* init basic PC hardware */

    pc_basic_device_init(isa_bus, gsi, &rtc_state, &floppy, xen_enabled());



    pc_nic_init(isa_bus, pci_bus);



    ide_drive_get(hd, MAX_IDE_BUS);

    if (pci_enabled) {

        PCIDevice *dev;

        if (xen_enabled()) {

            dev = pci_piix3_xen_ide_init(pci_bus, hd, piix3_devfn + 1);

        } else {

            dev = pci_piix3_ide_init(pci_bus, hd, piix3_devfn + 1);

        }

        idebus[0] = qdev_get_child_bus(&dev->qdev, "ide.0");

        idebus[1] = qdev_get_child_bus(&dev->qdev, "ide.1");

    } else {

        for(i = 0; i < MAX_IDE_BUS; i++) {

            ISADevice *dev;

            dev = isa_ide_init(isa_bus, ide_iobase[i], ide_iobase2[i],

                               ide_irq[i],

                               hd[MAX_IDE_DEVS * i], hd[MAX_IDE_DEVS * i + 1]);

            idebus[i] = qdev_get_child_bus(DEVICE(dev), "ide.0");

        }

    }



    pc_cmos_init(below_4g_mem_size, above_4g_mem_size, boot_device,

                 floppy, idebus[0], idebus[1], rtc_state);



    if (pci_enabled && usb_enabled(false)) {

        pci_create_simple(pci_bus, piix3_devfn + 2, "piix3-usb-uhci");

    }



    if (pci_enabled && acpi_enabled) {

        i2c_bus *smbus;



        smi_irq = qemu_allocate_irqs(pc_acpi_smi_interrupt,

                                     x86_env_get_cpu(first_cpu), 1);

        /* TODO: Populate SPD eeprom data.  */

        smbus = piix4_pm_init(pci_bus, piix3_devfn + 3, 0xb100,

                              gsi[9], *smi_irq,

                              kvm_enabled(), fw_cfg);

        smbus_eeprom_init(smbus, 8, NULL, 0);

    }



    if (pci_enabled) {

        pc_pci_device_init(pci_bus);

    }



    if (has_pvpanic) {

        pvpanic_init(isa_bus);

    }

}

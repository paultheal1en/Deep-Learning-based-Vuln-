static void pc_init1(ram_addr_t ram_size,

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

    PCII440FXState *i440fx_state;

    int piix3_devfn = -1;

    qemu_irq *cpu_irq;

    qemu_irq *isa_irq;

    qemu_irq *i8259;

    qemu_irq *cmos_s3;

    qemu_irq *smi_irq;

    IsaIrqState *isa_irq_state;

    DriveInfo *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    FDCtrl *floppy_controller;

    BusState *idebus[MAX_IDE_BUS];

    ISADevice *rtc_state;



    pc_cpus_init(cpu_model);



    if (kvmclock_enabled) {

        kvmclock_create();

    }



    /* allocate ram and load rom/bios */

    pc_memory_init(ram_size, kernel_filename, kernel_cmdline, initrd_filename,

                   &below_4g_mem_size, &above_4g_mem_size);



    cpu_irq = pc_allocate_cpu_irq();

    i8259 = i8259_init(cpu_irq[0]);

    isa_irq_state = qemu_mallocz(sizeof(*isa_irq_state));

    isa_irq_state->i8259 = i8259;

    if (pci_enabled) {

        ioapic_init(isa_irq_state);

    }

    isa_irq = qemu_allocate_irqs(isa_irq_handler, isa_irq_state, 24);



    if (pci_enabled) {

        pci_bus = i440fx_init(&i440fx_state, &piix3_devfn, isa_irq, ram_size);

    } else {

        pci_bus = NULL;

        i440fx_state = NULL;

        isa_bus_new(NULL);

    }

    isa_bus_irqs(isa_irq);



    pc_register_ferr_irq(isa_reserve_irq(13));



    pc_vga_init(pci_enabled? pci_bus: NULL);



    /* init basic PC hardware */

    pc_basic_device_init(isa_irq, &floppy_controller, &rtc_state);



    for(i = 0; i < nb_nics; i++) {

        NICInfo *nd = &nd_table[i];



        if (!pci_enabled || (nd->model && strcmp(nd->model, "ne2k_isa") == 0))

            pc_init_ne2k_isa(nd);

        else

            pci_nic_init_nofail(nd, "e1000", NULL);

    }



    if (drive_get_max_bus(IF_IDE) >= MAX_IDE_BUS) {

        fprintf(stderr, "qemu: too many IDE bus\n");

        exit(1);

    }



    for(i = 0; i < MAX_IDE_BUS * MAX_IDE_DEVS; i++) {

        hd[i] = drive_get(IF_IDE, i / MAX_IDE_DEVS, i % MAX_IDE_DEVS);

    }



    if (pci_enabled) {

        PCIDevice *dev;

        dev = pci_piix3_ide_init(pci_bus, hd, piix3_devfn + 1);

        idebus[0] = qdev_get_child_bus(&dev->qdev, "ide.0");

        idebus[1] = qdev_get_child_bus(&dev->qdev, "ide.1");

    } else {

        for(i = 0; i < MAX_IDE_BUS; i++) {

            ISADevice *dev;

            dev = isa_ide_init(ide_iobase[i], ide_iobase2[i], ide_irq[i],

                               hd[MAX_IDE_DEVS * i], hd[MAX_IDE_DEVS * i + 1]);

            idebus[i] = qdev_get_child_bus(&dev->qdev, "ide.0");

        }

    }



    audio_init(isa_irq, pci_enabled ? pci_bus : NULL);



    pc_cmos_init(below_4g_mem_size, above_4g_mem_size, boot_device,

                 idebus[0], idebus[1], floppy_controller, rtc_state);



    if (pci_enabled && usb_enabled) {

        usb_uhci_piix3_init(pci_bus, piix3_devfn + 2);

    }



    if (pci_enabled && acpi_enabled) {

        uint8_t *eeprom_buf = qemu_mallocz(8 * 256); /* XXX: make this persistent */

        i2c_bus *smbus;



        cmos_s3 = qemu_allocate_irqs(pc_cmos_set_s3_resume, rtc_state, 1);

        smi_irq = qemu_allocate_irqs(pc_acpi_smi_interrupt, first_cpu, 1);

        /* TODO: Populate SPD eeprom data.  */

        smbus = piix4_pm_init(pci_bus, piix3_devfn + 3, 0xb100,

                              isa_reserve_irq(9), *cmos_s3, *smi_irq,

                              kvm_enabled());

        for (i = 0; i < 8; i++) {

            DeviceState *eeprom;

            eeprom = qdev_create((BusState *)smbus, "smbus-eeprom");

            qdev_prop_set_uint8(eeprom, "address", 0x50 + i);

            qdev_prop_set_ptr(eeprom, "data", eeprom_buf + (i * 256));

            qdev_init_nofail(eeprom);

        }

    }



    if (i440fx_state) {

        i440fx_init_memory_mappings(i440fx_state);

    }



    if (pci_enabled) {

        pc_pci_device_init(pci_bus);

    }

}

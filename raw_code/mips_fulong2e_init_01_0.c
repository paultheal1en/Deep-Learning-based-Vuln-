static void mips_fulong2e_init(ram_addr_t ram_size, const char *boot_device,

                        const char *kernel_filename, const char *kernel_cmdline,

                        const char *initrd_filename, const char *cpu_model)

{

    char *filename;

    unsigned long ram_offset, bios_offset;

    long bios_size;

    int64_t kernel_entry;

    qemu_irq *i8259;

    qemu_irq *cpu_exit_irq;

    int via_devfn;

    PCIBus *pci_bus;

    uint8_t *eeprom_buf;

    i2c_bus *smbus;

    int i;

    DriveInfo *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    DeviceState *eeprom;

    CPUState *env;



    /* init CPUs */

    if (cpu_model == NULL) {

        cpu_model = "Loongson-2E";

    }

    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }



    register_savevm(NULL, "cpu", 0, 3, cpu_save, cpu_load, env);

    qemu_register_reset(main_cpu_reset, env);



    /* fulong 2e has 256M ram. */

    ram_size = 256 * 1024 * 1024;



    /* fulong 2e has a 1M flash.Winbond W39L040AP70Z */

    bios_size = 1024 * 1024;



    /* allocate RAM */

    ram_offset = qemu_ram_alloc(NULL, "fulong2e.ram", ram_size);

    bios_offset = qemu_ram_alloc(NULL, "fulong2e.bios", bios_size);



    cpu_register_physical_memory(0, ram_size, ram_offset);

    cpu_register_physical_memory(0x1fc00000LL,

					   bios_size, bios_offset | IO_MEM_ROM);



    /* We do not support flash operation, just loading pmon.bin as raw BIOS.

     * Please use -L to set the BIOS path and -bios to set bios name. */



    if (kernel_filename) {

        loaderparams.ram_size = ram_size;

        loaderparams.kernel_filename = kernel_filename;

        loaderparams.kernel_cmdline = kernel_cmdline;

        loaderparams.initrd_filename = initrd_filename;

        kernel_entry = load_kernel (env);

        write_bootloader(env, qemu_get_ram_ptr(bios_offset), kernel_entry);

    } else {

        if (bios_name == NULL) {

                bios_name = FULONG_BIOSNAME;

        }

        filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

        if (filename) {

            bios_size = load_image_targphys(filename, 0x1fc00000LL,

                                            BIOS_SIZE);

            qemu_free(filename);

        } else {

            bios_size = -1;

        }



        if ((bios_size < 0 || bios_size > BIOS_SIZE) && !kernel_filename) {

            fprintf(stderr, "qemu: Could not load MIPS bios '%s'\n", bios_name);

            exit(1);

        }

    }



    /* Init internal devices */

    cpu_mips_irq_init_cpu(env);

    cpu_mips_clock_init(env);



    /* Interrupt controller */

    /* The 8259 -> IP5  */

    i8259 = i8259_init(env->irq[5]);



    /* North bridge, Bonito --> IP2 */

    pci_bus = bonito_init((qemu_irq *)&(env->irq[2]));



    /* South bridge */

    if (drive_get_max_bus(IF_IDE) >= MAX_IDE_BUS) {

        fprintf(stderr, "qemu: too many IDE bus\n");

        exit(1);

    }



    for(i = 0; i < MAX_IDE_BUS * MAX_IDE_DEVS; i++) {

        hd[i] = drive_get(IF_IDE, i / MAX_IDE_DEVS, i % MAX_IDE_DEVS);

    }



    via_devfn = vt82c686b_init(pci_bus, PCI_DEVFN(FULONG2E_VIA_SLOT, 0));

    if (via_devfn < 0) {

        fprintf(stderr, "vt82c686b_init error \n");

        exit(1);

    }



    isa_bus_irqs(i8259);

    vt82c686b_ide_init(pci_bus, hd, PCI_DEVFN(FULONG2E_VIA_SLOT, 1));

    usb_uhci_vt82c686b_init(pci_bus, PCI_DEVFN(FULONG2E_VIA_SLOT, 2));

    usb_uhci_vt82c686b_init(pci_bus, PCI_DEVFN(FULONG2E_VIA_SLOT, 3));



    smbus = vt82c686b_pm_init(pci_bus, PCI_DEVFN(FULONG2E_VIA_SLOT, 4),

                              0xeee1, NULL);

    eeprom_buf = qemu_mallocz(8 * 256); /* XXX: make this persistent */

    memcpy(eeprom_buf, eeprom_spd, sizeof(eeprom_spd));

    /* TODO: Populate SPD eeprom data.  */

    eeprom = qdev_create((BusState *)smbus, "smbus-eeprom");

    qdev_prop_set_uint8(eeprom, "address", 0x50);

    qdev_prop_set_ptr(eeprom, "data", eeprom_buf);

    qdev_init_nofail(eeprom);



    /* init other devices */

    pit = pit_init(0x40, isa_reserve_irq(0));

    cpu_exit_irq = qemu_allocate_irqs(cpu_request_exit, NULL, 1);

    DMA_init(0, cpu_exit_irq);



    /* Super I/O */

    isa_create_simple("i8042");



    rtc_init(2000, NULL);



    for(i = 0; i < MAX_SERIAL_PORTS; i++) {

        if (serial_hds[i]) {

            serial_isa_init(i, serial_hds[i]);

        }

    }



    if (parallel_hds[0]) {

        parallel_init(0, parallel_hds[0]);

    }



    /* Sound card */

    audio_init(pci_bus);

    /* Network card */

    network_init();

}

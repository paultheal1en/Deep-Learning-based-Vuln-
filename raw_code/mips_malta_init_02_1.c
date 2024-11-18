void mips_malta_init (ram_addr_t ram_size,

                      const char *boot_device,

                      const char *kernel_filename, const char *kernel_cmdline,

                      const char *initrd_filename, const char *cpu_model)

{

    char *filename;

    ram_addr_t ram_offset;

    ram_addr_t bios_offset;

    target_long bios_size;

    int64_t kernel_entry;

    PCIBus *pci_bus;

    ISADevice *isa_dev;

    CPUState *env;

    RTCState *rtc_state;

    fdctrl_t *floppy_controller;

    MaltaFPGAState *malta_fpga;

    qemu_irq *i8259;

    int piix4_devfn;

    uint8_t *eeprom_buf;

    i2c_bus *smbus;

    int i;

    DriveInfo *dinfo;

    DriveInfo *hd[MAX_IDE_BUS * MAX_IDE_DEVS];

    DriveInfo *fd[MAX_FD];

    int fl_idx = 0;

    int fl_sectors = 0;



    /* Make sure the first 3 serial ports are associated with a device. */

    for(i = 0; i < 3; i++) {

        if (!serial_hds[i]) {

            char label[32];

            snprintf(label, sizeof(label), "serial%d", i);

            serial_hds[i] = qemu_chr_open(label, "null", NULL);

        }

    }



    /* init CPUs */

    if (cpu_model == NULL) {

#ifdef TARGET_MIPS64

        cpu_model = "20Kc";

#else

        cpu_model = "24Kf";

#endif

    }

    env = cpu_init(cpu_model);

    if (!env) {

        fprintf(stderr, "Unable to find CPU definition\n");

        exit(1);

    }

    qemu_register_reset(main_cpu_reset, env);



    /* allocate RAM */

    if (ram_size > (256 << 20)) {

        fprintf(stderr,

                "qemu: Too much memory for this machine: %d MB, maximum 256 MB\n",

                ((unsigned int)ram_size / (1 << 20)));

        exit(1);

    }

    ram_offset = qemu_ram_alloc(ram_size);

    bios_offset = qemu_ram_alloc(BIOS_SIZE);





    cpu_register_physical_memory(0, ram_size, ram_offset | IO_MEM_RAM);



    /* Map the bios at two physical locations, as on the real board. */

    cpu_register_physical_memory(0x1e000000LL,

                                 BIOS_SIZE, bios_offset | IO_MEM_ROM);

    cpu_register_physical_memory(0x1fc00000LL,

                                 BIOS_SIZE, bios_offset | IO_MEM_ROM);



    /* FPGA */

    malta_fpga = malta_fpga_init(0x1f000000LL, env->irq[2], serial_hds[2]);



    /* Load firmware in flash / BIOS unless we boot directly into a kernel. */

    if (kernel_filename) {

        /* Write a small bootloader to the flash location. */

        loaderparams.ram_size = ram_size;

        loaderparams.kernel_filename = kernel_filename;

        loaderparams.kernel_cmdline = kernel_cmdline;

        loaderparams.initrd_filename = initrd_filename;

        kernel_entry = load_kernel(env);

        env->CP0_Status &= ~((1 << CP0St_BEV) | (1 << CP0St_ERL));

        write_bootloader(env, qemu_get_ram_ptr(bios_offset), kernel_entry);

    } else {

        dinfo = drive_get(IF_PFLASH, 0, fl_idx);

        if (dinfo) {

            /* Load firmware from flash. */

            bios_size = 0x400000;

            fl_sectors = bios_size >> 16;

#ifdef DEBUG_BOARD_INIT

            printf("Register parallel flash %d size " TARGET_FMT_lx " at "

                   "offset %08lx addr %08llx '%s' %x\n",

                   fl_idx, bios_size, bios_offset, 0x1e000000LL,

                   bdrv_get_device_name(dinfo->bdrv), fl_sectors);

#endif

            pflash_cfi01_register(0x1e000000LL, bios_offset,

                                  dinfo->bdrv, 65536, fl_sectors,

                                  4, 0x0000, 0x0000, 0x0000, 0x0000);

            fl_idx++;

        } else {

            /* Load a BIOS image. */

            if (bios_name == NULL)

                bios_name = BIOS_FILENAME;

            filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, bios_name);

            if (filename) {

                bios_size = load_image_targphys(filename, 0x1fc00000LL,

                                                BIOS_SIZE);

                qemu_free(filename);

            } else {

                bios_size = -1;

            }

            if ((bios_size < 0 || bios_size > BIOS_SIZE) && !kernel_filename) {

                fprintf(stderr,

                        "qemu: Could not load MIPS bios '%s', and no -kernel argument was specified\n",

                        bios_name);

                exit(1);

            }

        }

        /* In little endian mode the 32bit words in the bios are swapped,

           a neat trick which allows bi-endian firmware. */

#ifndef TARGET_WORDS_BIGENDIAN

        {

            uint32_t *addr = qemu_get_ram_ptr(bios_offset);;

            uint32_t *end = addr + bios_size;

            while (addr < end) {

                bswap32s(addr);

            }

        }

#endif

    }



    /* Board ID = 0x420 (Malta Board with CoreLV)

       XXX: theoretically 0x1e000010 should map to flash and 0x1fc00010 should

       map to the board ID. */

    stl_phys(0x1fc00010LL, 0x00000420);



    /* Init internal devices */

    cpu_mips_irq_init_cpu(env);

    cpu_mips_clock_init(env);



    /* Interrupt controller */

    /* The 8259 is attached to the MIPS CPU INT0 pin, ie interrupt 2 */

    i8259 = i8259_init(env->irq[2]);



    /* Northbridge */

    pci_bus = pci_gt64120_init(i8259);



    /* Southbridge */



    if (drive_get_max_bus(IF_IDE) >= MAX_IDE_BUS) {

        fprintf(stderr, "qemu: too many IDE bus\n");

        exit(1);

    }



    for(i = 0; i < MAX_IDE_BUS * MAX_IDE_DEVS; i++) {

        hd[i] = drive_get(IF_IDE, i / MAX_IDE_DEVS, i % MAX_IDE_DEVS);

    }



    piix4_devfn = piix4_init(pci_bus, 80);

    isa_bus_irqs(i8259);

    pci_piix4_ide_init(pci_bus, hd, piix4_devfn + 1);

    usb_uhci_piix4_init(pci_bus, piix4_devfn + 2);

    smbus = piix4_pm_init(pci_bus, piix4_devfn + 3, 0x1100, isa_reserve_irq(9));

    eeprom_buf = qemu_mallocz(8 * 256); /* XXX: make this persistent */

    for (i = 0; i < 8; i++) {

        /* TODO: Populate SPD eeprom data.  */

        DeviceState *eeprom;

        eeprom = qdev_create((BusState *)smbus, "smbus-eeprom");

        qdev_prop_set_uint8(eeprom, "address", 0x50 + i);

        qdev_prop_set_ptr(eeprom, "data", eeprom_buf + (i * 256));

        qdev_init(eeprom);

    }

    pit = pit_init(0x40, isa_reserve_irq(0));

    DMA_init(0);



    /* Super I/O */

    isa_dev = isa_create_simple("i8042");

 

    rtc_state = rtc_init(2000);

    serial_isa_init(0, serial_hds[0]);

    serial_isa_init(1, serial_hds[1]);

    if (parallel_hds[0])

        parallel_init(0, parallel_hds[0]);

    for(i = 0; i < MAX_FD; i++) {

        fd[i] = drive_get(IF_FLOPPY, 0, i);

    }

    floppy_controller = fdctrl_init_isa(fd);



    /* Sound card */

#ifdef HAS_AUDIO

    audio_init(pci_bus);

#endif



    /* Network card */

    network_init();



    /* Optional PCI video card */

    if (cirrus_vga_enabled) {

        pci_cirrus_vga_init(pci_bus);

    } else if (vmsvga_enabled) {

        pci_vmsvga_init(pci_bus);

    } else if (std_vga_enabled) {

        pci_vga_init(pci_bus, 0, 0);

    }

}

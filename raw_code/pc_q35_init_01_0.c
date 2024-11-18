static void pc_q35_init(MachineState *machine)

{

    PCMachineState *pcms = PC_MACHINE(machine);

    Q35PCIHost *q35_host;

    PCIHostState *phb;

    PCIBus *host_bus;

    PCIDevice *lpc;

    BusState *idebus[MAX_SATA_PORTS];

    ISADevice *rtc_state;

    MemoryRegion *pci_memory;

    MemoryRegion *rom_memory;

    MemoryRegion *ram_memory;

    GSIState *gsi_state;

    ISABus *isa_bus;

    int pci_enabled = 1;

    qemu_irq *gsi;

    qemu_irq *i8259;

    int i;

    ICH9LPCState *ich9_lpc;

    PCIDevice *ahci;

    DeviceState *icc_bridge;

    PcGuestInfo *guest_info;

    ram_addr_t lowmem;

    DriveInfo *hd[MAX_SATA_PORTS];

    MachineClass *mc = MACHINE_GET_CLASS(machine);



    /* Check whether RAM fits below 4G (leaving 1/2 GByte for IO memory

     * and 256 Mbytes for PCI Express Enhanced Configuration Access Mapping

     * also known as MMCFG).

     * If it doesn't, we need to split it in chunks below and above 4G.

     * In any case, try to make sure that guest addresses aligned at

     * 1G boundaries get mapped to host addresses aligned at 1G boundaries.

     * For old machine types, use whatever split we used historically to avoid

     * breaking migration.

     */

    if (machine->ram_size >= 0xb0000000) {

        lowmem = gigabyte_align ? 0x80000000 : 0xb0000000;

    } else {

        lowmem = 0xb0000000;

    }



    /* Handle the machine opt max-ram-below-4g.  It is basically doing

     * min(qemu limit, user limit).

     */

    if (lowmem > pcms->max_ram_below_4g) {

        lowmem = pcms->max_ram_below_4g;

        if (machine->ram_size - lowmem > lowmem &&

            lowmem & ((1ULL << 30) - 1)) {

            error_report("Warning: Large machine and max_ram_below_4g(%"PRIu64

                         ") not a multiple of 1G; possible bad performance.",

                         pcms->max_ram_below_4g);

        }

    }



    if (machine->ram_size >= lowmem) {

        pcms->above_4g_mem_size = machine->ram_size - lowmem;

        pcms->below_4g_mem_size = lowmem;

    } else {

        pcms->above_4g_mem_size = 0;

        pcms->below_4g_mem_size = machine->ram_size;

    }



    if (xen_enabled() && xen_hvm_init(pcms, &ram_memory) != 0) {

        fprintf(stderr, "xen hardware virtual machine initialisation failed\n");

        exit(1);

    }



    icc_bridge = qdev_create(NULL, TYPE_ICC_BRIDGE);

    object_property_add_child(qdev_get_machine(), "icc-bridge",

                              OBJECT(icc_bridge), NULL);



    pc_cpus_init(machine->cpu_model, icc_bridge);

    pc_acpi_init("q35-acpi-dsdt.aml");



    kvmclock_create();



    /* pci enabled */

    if (pci_enabled) {

        pci_memory = g_new(MemoryRegion, 1);

        memory_region_init(pci_memory, NULL, "pci", UINT64_MAX);

        rom_memory = pci_memory;

    } else {

        pci_memory = NULL;

        rom_memory = get_system_memory();

    }



    guest_info = pc_guest_info_init(pcms);

    guest_info->isapc_ram_fw = false;

    guest_info->has_acpi_build = has_acpi_build;

    guest_info->has_reserved_memory = has_reserved_memory;

    guest_info->rsdp_in_ram = rsdp_in_ram;



    /* Migration was not supported in 2.0 for Q35, so do not bother

     * with this hack (see hw/i386/acpi-build.c).

     */

    guest_info->legacy_acpi_table_size = 0;



    if (smbios_defaults) {

        /* These values are guest ABI, do not change */

        smbios_set_defaults("QEMU", "Standard PC (Q35 + ICH9, 2009)",

                            mc->name, smbios_legacy_mode, smbios_uuid_encoded,

                            SMBIOS_ENTRY_POINT_21);

    }



    /* allocate ram and load rom/bios */

    if (!xen_enabled()) {

        pc_memory_init(pcms, get_system_memory(),

                       rom_memory, &ram_memory, guest_info);

    }



    /* irq lines */

    gsi_state = g_malloc0(sizeof(*gsi_state));

    if (kvm_irqchip_in_kernel()) {

        kvm_pc_setup_irq_routing(pci_enabled);

        gsi = qemu_allocate_irqs(kvm_pc_gsi_handler, gsi_state,

                                 GSI_NUM_PINS);

    } else {

        gsi = qemu_allocate_irqs(gsi_handler, gsi_state, GSI_NUM_PINS);

    }



    /* create pci host bus */

    q35_host = Q35_HOST_DEVICE(qdev_create(NULL, TYPE_Q35_HOST_DEVICE));



    object_property_add_child(qdev_get_machine(), "q35", OBJECT(q35_host), NULL);

    q35_host->mch.ram_memory = ram_memory;

    q35_host->mch.pci_address_space = pci_memory;

    q35_host->mch.system_memory = get_system_memory();

    q35_host->mch.address_space_io = get_system_io();

    q35_host->mch.below_4g_mem_size = pcms->below_4g_mem_size;

    q35_host->mch.above_4g_mem_size = pcms->above_4g_mem_size;

    q35_host->mch.guest_info = guest_info;

    /* pci */

    qdev_init_nofail(DEVICE(q35_host));

    phb = PCI_HOST_BRIDGE(q35_host);

    host_bus = phb->bus;

    /* create ISA bus */

    lpc = pci_create_simple_multifunction(host_bus, PCI_DEVFN(ICH9_LPC_DEV,

                                          ICH9_LPC_FUNC), true,

                                          TYPE_ICH9_LPC_DEVICE);



    object_property_add_link(OBJECT(machine), PC_MACHINE_ACPI_DEVICE_PROP,

                             TYPE_HOTPLUG_HANDLER,

                             (Object **)&pcms->acpi_dev,

                             object_property_allow_set_link,

                             OBJ_PROP_LINK_UNREF_ON_RELEASE, &error_abort);

    object_property_set_link(OBJECT(machine), OBJECT(lpc),

                             PC_MACHINE_ACPI_DEVICE_PROP, &error_abort);



    ich9_lpc = ICH9_LPC_DEVICE(lpc);

    ich9_lpc->pic = gsi;

    ich9_lpc->ioapic = gsi_state->ioapic_irq;

    pci_bus_irqs(host_bus, ich9_lpc_set_irq, ich9_lpc_map_irq, ich9_lpc,

                 ICH9_LPC_NB_PIRQS);

    pci_bus_set_route_irq_fn(host_bus, ich9_route_intx_pin_to_irq);

    isa_bus = ich9_lpc->isa_bus;



    /*end early*/

    isa_bus_irqs(isa_bus, gsi);



    if (kvm_irqchip_in_kernel()) {

        i8259 = kvm_i8259_init(isa_bus);

    } else if (xen_enabled()) {

        i8259 = xen_interrupt_controller_init();

    } else {

        i8259 = i8259_init(isa_bus, pc_allocate_cpu_irq());

    }



    for (i = 0; i < ISA_NUM_IRQS; i++) {

        gsi_state->i8259_irq[i] = i8259[i];

    }

    if (pci_enabled) {

        ioapic_init_gsi(gsi_state, "q35");

    }

    qdev_init_nofail(icc_bridge);



    pc_register_ferr_irq(gsi[13]);



    assert(pcms->vmport != ON_OFF_AUTO_MAX);

    if (pcms->vmport == ON_OFF_AUTO_AUTO) {

        pcms->vmport = xen_enabled() ? ON_OFF_AUTO_OFF : ON_OFF_AUTO_ON;

    }



    /* init basic PC hardware */

    pc_basic_device_init(isa_bus, gsi, &rtc_state, !mc->no_floppy,

                         (pcms->vmport != ON_OFF_AUTO_ON), 0xff0104);



    /* connect pm stuff to lpc */

    ich9_lpc_pm_init(lpc, pc_machine_is_smm_enabled(pcms), !mc->no_tco);



    /* ahci and SATA device, for q35 1 ahci controller is built-in */

    ahci = pci_create_simple_multifunction(host_bus,

                                           PCI_DEVFN(ICH9_SATA1_DEV,

                                                     ICH9_SATA1_FUNC),

                                           true, "ich9-ahci");

    idebus[0] = qdev_get_child_bus(&ahci->qdev, "ide.0");

    idebus[1] = qdev_get_child_bus(&ahci->qdev, "ide.1");

    g_assert(MAX_SATA_PORTS == ICH_AHCI(ahci)->ahci.ports);

    ide_drive_get(hd, ICH_AHCI(ahci)->ahci.ports);

    ahci_ide_create_devs(ahci, hd);



    if (usb_enabled()) {

        /* Should we create 6 UHCI according to ich9 spec? */

        ehci_create_ich9_with_companions(host_bus, 0x1d);

    }



    /* TODO: Populate SPD eeprom data.  */

    smbus_eeprom_init(ich9_smb_init(host_bus,

                                    PCI_DEVFN(ICH9_SMB_DEV, ICH9_SMB_FUNC),

                                    0xb100),

                      8, NULL, 0);



    pc_cmos_init(pcms, idebus[0], idebus[1], rtc_state);



    /* the rest devices to which pci devfn is automatically assigned */

    pc_vga_init(isa_bus, host_bus);

    pc_nic_init(isa_bus, host_bus);

    if (pci_enabled) {

        pc_pci_device_init(host_bus);

    }

}

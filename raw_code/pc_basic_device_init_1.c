void pc_basic_device_init(ISABus *isa_bus, qemu_irq *gsi,

                          ISADevice **rtc_state,

                          bool create_fdctrl,

                          bool no_vmport,

                          uint32_t hpet_irqs)

{

    int i;

    DriveInfo *fd[MAX_FD];

    DeviceState *hpet = NULL;

    int pit_isa_irq = 0;

    qemu_irq pit_alt_irq = NULL;

    qemu_irq rtc_irq = NULL;

    qemu_irq *a20_line;

    ISADevice *i8042, *port92, *vmmouse, *pit = NULL;

    MemoryRegion *ioport80_io = g_new(MemoryRegion, 1);

    MemoryRegion *ioportF0_io = g_new(MemoryRegion, 1);



    memory_region_init_io(ioport80_io, NULL, &ioport80_io_ops, NULL, "ioport80", 1);

    memory_region_add_subregion(isa_bus->address_space_io, 0x80, ioport80_io);



    memory_region_init_io(ioportF0_io, NULL, &ioportF0_io_ops, NULL, "ioportF0", 1);

    memory_region_add_subregion(isa_bus->address_space_io, 0xf0, ioportF0_io);



    /*

     * Check if an HPET shall be created.

     *

     * Without KVM_CAP_PIT_STATE2, we cannot switch off the in-kernel PIT

     * when the HPET wants to take over. Thus we have to disable the latter.

     */

    if (!no_hpet && (!kvm_irqchip_in_kernel() || kvm_has_pit_state2())) {

        /* In order to set property, here not using sysbus_try_create_simple */

        hpet = qdev_try_create(NULL, TYPE_HPET);

        if (hpet) {

            /* For pc-piix-*, hpet's intcap is always IRQ2. For pc-q35-1.7

             * and earlier, use IRQ2 for compat. Otherwise, use IRQ16~23,

             * IRQ8 and IRQ2.

             */

            uint8_t compat = object_property_get_int(OBJECT(hpet),

                    HPET_INTCAP, NULL);

            if (!compat) {

                qdev_prop_set_uint32(hpet, HPET_INTCAP, hpet_irqs);

            }

            qdev_init_nofail(hpet);

            sysbus_mmio_map(SYS_BUS_DEVICE(hpet), 0, HPET_BASE);



            for (i = 0; i < GSI_NUM_PINS; i++) {

                sysbus_connect_irq(SYS_BUS_DEVICE(hpet), i, gsi[i]);

            }

            pit_isa_irq = -1;

            pit_alt_irq = qdev_get_gpio_in(hpet, HPET_LEGACY_PIT_INT);

            rtc_irq = qdev_get_gpio_in(hpet, HPET_LEGACY_RTC_INT);

        }

    }

    *rtc_state = rtc_init(isa_bus, 2000, rtc_irq);



    qemu_register_boot_set(pc_boot_set, *rtc_state);



    if (!xen_enabled()) {

        if (kvm_pit_in_kernel()) {

            pit = kvm_pit_init(isa_bus, 0x40);

        } else {

            pit = pit_init(isa_bus, 0x40, pit_isa_irq, pit_alt_irq);

        }

        if (hpet) {

            /* connect PIT to output control line of the HPET */

            qdev_connect_gpio_out(hpet, 0, qdev_get_gpio_in(DEVICE(pit), 0));

        }

        pcspk_init(isa_bus, pit);

    }



    serial_hds_isa_init(isa_bus, MAX_SERIAL_PORTS);

    parallel_hds_isa_init(isa_bus, MAX_PARALLEL_PORTS);



    a20_line = qemu_allocate_irqs(handle_a20_line_change, first_cpu, 2);

    i8042 = isa_create_simple(isa_bus, "i8042");

    i8042_setup_a20_line(i8042, a20_line[0]);

    if (!no_vmport) {

        vmport_init(isa_bus);

        vmmouse = isa_try_create(isa_bus, "vmmouse");

    } else {

        vmmouse = NULL;

    }

    if (vmmouse) {

        DeviceState *dev = DEVICE(vmmouse);

        qdev_prop_set_ptr(dev, "ps2_mouse", i8042);

        qdev_init_nofail(dev);

    }

    port92 = isa_create_simple(isa_bus, "port92");

    port92_init(port92, a20_line[1]);




    DMA_init(isa_bus, 0);



    for(i = 0; i < MAX_FD; i++) {

        fd[i] = drive_get(IF_FLOPPY, 0, i);

        create_fdctrl |= !!fd[i];

    }

    if (create_fdctrl) {

        fdctrl_init_isa(isa_bus, fd);

    }

}
static void build_pci_bus_end(PCIBus *bus, void *bus_state)

{

    AcpiBuildPciBusHotplugState *child = bus_state;

    AcpiBuildPciBusHotplugState *parent = child->parent;

    GArray *bus_table = build_alloc_array();

    DECLARE_BITMAP(slot_hotplug_enable, PCI_SLOT_MAX);

    DECLARE_BITMAP(slot_device_present, PCI_SLOT_MAX);

    DECLARE_BITMAP(slot_device_system, PCI_SLOT_MAX);

    DECLARE_BITMAP(slot_device_vga, PCI_SLOT_MAX);

    DECLARE_BITMAP(slot_device_qxl, PCI_SLOT_MAX);

    uint8_t op;

    int i;

    QObject *bsel;

    GArray *method;

    bool bus_hotplug_support = false;



    if (bus->parent_dev) {

        op = 0x82; /* DeviceOp */

        build_append_nameseg(bus_table, "S%.02X_",

                             bus->parent_dev->devfn);

        build_append_byte(bus_table, 0x08); /* NameOp */

        build_append_nameseg(bus_table, "_SUN");

        build_append_value(bus_table, PCI_SLOT(bus->parent_dev->devfn), 1);

        build_append_byte(bus_table, 0x08); /* NameOp */

        build_append_nameseg(bus_table, "_ADR");

        build_append_value(bus_table, (PCI_SLOT(bus->parent_dev->devfn) << 16) |

                           PCI_FUNC(bus->parent_dev->devfn), 4);

    } else {

        op = 0x10; /* ScopeOp */;

        build_append_nameseg(bus_table, "PCI0");

    }



    bsel = object_property_get_qobject(OBJECT(bus), ACPI_PCIHP_PROP_BSEL, NULL);

    if (bsel) {

        build_append_byte(bus_table, 0x08); /* NameOp */

        build_append_nameseg(bus_table, "BSEL");

        build_append_int(bus_table, qint_get_int(qobject_to_qint(bsel)));

        memset(slot_hotplug_enable, 0xff, sizeof slot_hotplug_enable);

    } else {

        /* No bsel - no slots are hot-pluggable */

        memset(slot_hotplug_enable, 0x00, sizeof slot_hotplug_enable);

    }



    memset(slot_device_present, 0x00, sizeof slot_device_present);

    memset(slot_device_system, 0x00, sizeof slot_device_present);

    memset(slot_device_vga, 0x00, sizeof slot_device_vga);

    memset(slot_device_qxl, 0x00, sizeof slot_device_qxl);



    for (i = 0; i < ARRAY_SIZE(bus->devices); i += PCI_FUNC_MAX) {

        DeviceClass *dc;

        PCIDeviceClass *pc;

        PCIDevice *pdev = bus->devices[i];

        int slot = PCI_SLOT(i);



        if (!pdev) {

            continue;

        }



        set_bit(slot, slot_device_present);

        pc = PCI_DEVICE_GET_CLASS(pdev);

        dc = DEVICE_GET_CLASS(pdev);



        if (pc->class_id == PCI_CLASS_BRIDGE_ISA) {

            set_bit(slot, slot_device_system);

        }



        if (pc->class_id == PCI_CLASS_DISPLAY_VGA) {

            set_bit(slot, slot_device_vga);



            if (object_dynamic_cast(OBJECT(pdev), "qxl-vga")) {

                set_bit(slot, slot_device_qxl);

            }

        }



        if (!dc->hotpluggable || pc->is_bridge) {

            clear_bit(slot, slot_hotplug_enable);

        }

    }



    /* Append Device object for each slot */

    for (i = 0; i < PCI_SLOT_MAX; i++) {

        bool can_eject = test_bit(i, slot_hotplug_enable);

        bool present = test_bit(i, slot_device_present);

        bool vga = test_bit(i, slot_device_vga);

        bool qxl = test_bit(i, slot_device_qxl);

        bool system = test_bit(i, slot_device_system);

        if (can_eject) {

            void *pcihp = acpi_data_push(bus_table,

                                         ACPI_PCIHP_SIZEOF);

            memcpy(pcihp, ACPI_PCIHP_AML, ACPI_PCIHP_SIZEOF);

            patch_pcihp(i, pcihp);

            bus_hotplug_support = true;

        } else if (qxl) {

            void *pcihp = acpi_data_push(bus_table,

                                         ACPI_PCIQXL_SIZEOF);

            memcpy(pcihp, ACPI_PCIQXL_AML, ACPI_PCIQXL_SIZEOF);

            patch_pciqxl(i, pcihp);

        } else if (vga) {

            void *pcihp = acpi_data_push(bus_table,

                                         ACPI_PCIVGA_SIZEOF);

            memcpy(pcihp, ACPI_PCIVGA_AML, ACPI_PCIVGA_SIZEOF);

            patch_pcivga(i, pcihp);

        } else if (system) {

            /* Nothing to do: system devices are in DSDT. */

        } else if (present) {

            void *pcihp = acpi_data_push(bus_table,

                                         ACPI_PCINOHP_SIZEOF);

            memcpy(pcihp, ACPI_PCINOHP_AML, ACPI_PCINOHP_SIZEOF);

            patch_pcinohp(i, pcihp);

        }

    }



    if (bsel) {

        method = build_alloc_method("DVNT", 2);



        for (i = 0; i < PCI_SLOT_MAX; i++) {

            GArray *notify;

            uint8_t op;



            if (!test_bit(i, slot_hotplug_enable)) {

                continue;

            }



            notify = build_alloc_array();

            op = 0xA0; /* IfOp */



            build_append_byte(notify, 0x7B); /* AndOp */

            build_append_byte(notify, 0x68); /* Arg0Op */

            build_append_int(notify, 0x1 << i);

            build_append_byte(notify, 0x00); /* NullName */

            build_append_byte(notify, 0x86); /* NotifyOp */

            build_append_nameseg(notify, "S%.02X_", PCI_DEVFN(i, 0));

            build_append_byte(notify, 0x69); /* Arg1Op */



            /* Pack it up */

            build_package(notify, op, 0);



            build_append_array(method, notify);



            build_free_array(notify);

        }



        build_append_and_cleanup_method(bus_table, method);

    }



    /* Append PCNT method to notify about events on local and child buses.

     * Add unconditionally for root since DSDT expects it.

     */

    if (bus_hotplug_support || child->notify_table->len || !bus->parent_dev) {

        method = build_alloc_method("PCNT", 0);



        /* If bus supports hotplug select it and notify about local events */

        if (bsel) {

            build_append_byte(method, 0x70); /* StoreOp */

            build_append_int(method, qint_get_int(qobject_to_qint(bsel)));

            build_append_nameseg(method, "BNUM");

            build_append_nameseg(method, "DVNT");

            build_append_nameseg(method, "PCIU");

            build_append_int(method, 1); /* Device Check */

            build_append_nameseg(method, "DVNT");

            build_append_nameseg(method, "PCID");

            build_append_int(method, 3); /* Eject Request */

        }



        /* Notify about child bus events in any case */

        build_append_array(method, child->notify_table);



        build_append_and_cleanup_method(bus_table, method);



        /* Append description of child buses */

        build_append_array(bus_table, child->device_table);



        /* Pack it up */

        if (bus->parent_dev) {

            build_extop_package(bus_table, op);

        } else {

            build_package(bus_table, op, 0);

        }



        /* Append our bus description to parent table */

        build_append_array(parent->device_table, bus_table);



        /* Also tell parent how to notify us, invoking PCNT method.

         * At the moment this is not needed for root as we have a single root.

         */

        if (bus->parent_dev) {

            build_append_byte(parent->notify_table, '^'); /* ParentPrefixChar */

            build_append_byte(parent->notify_table, 0x2E); /* DualNamePrefix */

            build_append_nameseg(parent->notify_table, "S%.02X_",

                                 bus->parent_dev->devfn);

            build_append_nameseg(parent->notify_table, "PCNT");

        }

    }



    build_free_array(bus_table);

    build_pci_bus_state_cleanup(child);

    g_free(child);

}

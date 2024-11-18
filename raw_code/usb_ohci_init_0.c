static void usb_ohci_init(OHCIState *ohci, DeviceState *dev,

                          int num_ports, int devfn,

                          qemu_irq irq, enum ohci_type type,

                          const char *name, uint32_t localmem_base)

{

    int i;



    if (usb_frame_time == 0) {

#ifdef OHCI_TIME_WARP

        usb_frame_time = get_ticks_per_sec();

        usb_bit_time = muldiv64(1, get_ticks_per_sec(), USB_HZ/1000);

#else

        usb_frame_time = muldiv64(1, get_ticks_per_sec(), 1000);

        if (get_ticks_per_sec() >= USB_HZ) {

            usb_bit_time = muldiv64(1, get_ticks_per_sec(), USB_HZ);

        } else {

            usb_bit_time = 1;

        }

#endif

        dprintf("usb-ohci: usb_bit_time=%" PRId64 " usb_frame_time=%" PRId64 "\n",

                usb_frame_time, usb_bit_time);

    }



    ohci->mem = cpu_register_io_memory(ohci_readfn, ohci_writefn, ohci);

    ohci->localmem_base = localmem_base;

    ohci->name = name;



    ohci->irq = irq;

    ohci->type = type;



    usb_bus_new(&ohci->bus, dev);

    ohci->num_ports = num_ports;

    for (i = 0; i < num_ports; i++) {

        usb_register_port(&ohci->bus, &ohci->rhport[i].port, ohci, i, ohci_attach);

    }



    ohci->async_td = 0;

    qemu_register_reset(ohci_reset, ohci);

    ohci_reset(ohci);

}

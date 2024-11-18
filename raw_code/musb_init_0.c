struct MUSBState *musb_init(qemu_irq *irqs)

{

    MUSBState *s = g_malloc0(sizeof(*s));

    int i;



    s->irqs = irqs;



    s->faddr = 0x00;

    s->power = MGC_M_POWER_HSENAB;

    s->tx_intr = 0x0000;

    s->rx_intr = 0x0000;

    s->tx_mask = 0xffff;

    s->rx_mask = 0xffff;

    s->intr = 0x00;

    s->mask = 0x06;

    s->idx = 0;



    /* TODO: _DW */

    s->ep[0].config = MGC_M_CONFIGDATA_SOFTCONE | MGC_M_CONFIGDATA_DYNFIFO;

    for (i = 0; i < 16; i ++) {

        s->ep[i].fifosize = 64;

        s->ep[i].maxp[0] = 0x40;

        s->ep[i].maxp[1] = 0x40;

        s->ep[i].musb = s;

        s->ep[i].epnum = i;

        usb_packet_init(&s->ep[i].packey[0].p);

        usb_packet_init(&s->ep[i].packey[1].p);

    }



    usb_bus_new(&s->bus, &musb_bus_ops, NULL /* FIXME */);

    usb_register_port(&s->bus, &s->port, s, 0, &musb_port_ops,

                      USB_SPEED_MASK_LOW | USB_SPEED_MASK_FULL);



    return s;

}

static void handle_port_owner_write(EHCIState *s, int port, uint32_t owner)

{

    USBDevice *dev = s->ports[port].dev;

    uint32_t *portsc = &s->portsc[port];

    uint32_t orig;



    if (s->companion_ports[port] == NULL)

        return;



    owner = owner & PORTSC_POWNER;

    orig  = *portsc & PORTSC_POWNER;



    if (!(owner ^ orig)) {

        return;

    }



    if (dev) {

        usb_attach(&s->ports[port], NULL);

    }



    *portsc &= ~PORTSC_POWNER;

    *portsc |= owner;



    if (dev) {

        usb_attach(&s->ports[port], dev);

    }

}

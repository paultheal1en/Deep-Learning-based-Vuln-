static void ehci_detach(USBPort *port)

{

    EHCIState *s = port->opaque;

    uint32_t *portsc = &s->portsc[port->index];

    const char *owner = (*portsc & PORTSC_POWNER) ? "comp" : "ehci";



    trace_usb_ehci_port_detach(port->index, owner);



    if (*portsc & PORTSC_POWNER) {

        USBPort *companion = s->companion_ports[port->index];

        companion->ops->detach(companion);

        companion->dev = NULL;

        /*

         * EHCI spec 4.2.2: "When a disconnect occurs... On the event,

         * the port ownership is returned immediately to the EHCI controller."

         */

        *portsc &= ~PORTSC_POWNER;

        return;

    }



    ehci_queues_rip_device(s, port->dev, 0);

    ehci_queues_rip_device(s, port->dev, 1);



    *portsc &= ~(PORTSC_CONNECT|PORTSC_PED);

    *portsc |= PORTSC_CSC;



    ehci_raise_irq(s, USBSTS_PCD);

}

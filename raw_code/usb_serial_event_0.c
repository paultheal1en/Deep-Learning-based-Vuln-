static void usb_serial_event(void *opaque, int event)

{

    USBSerialState *s = opaque;



    switch (event) {

        case CHR_EVENT_BREAK:

            s->event_trigger |= FTDI_BI;

            break;

        case CHR_EVENT_FOCUS:

            break;

        case CHR_EVENT_OPENED:

            usb_serial_reset(s);

            /* TODO: Reset USB port */

            break;

    }

}

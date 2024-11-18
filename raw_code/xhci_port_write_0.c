static void xhci_port_write(void *ptr, hwaddr reg,

                            uint64_t val, unsigned size)

{

    XHCIPort *port = ptr;

    uint32_t portsc, notify;



    trace_usb_xhci_port_write(port->portnr, reg, val);



    switch (reg) {

    case 0x00: /* PORTSC */

        /* write-1-to-start bits */

        if (val & PORTSC_PR) {

            xhci_port_reset(port);

            break;

        }



        portsc = port->portsc;

        notify = 0;

        /* write-1-to-clear bits*/

        portsc &= ~(val & (PORTSC_CSC|PORTSC_PEC|PORTSC_WRC|PORTSC_OCC|

                           PORTSC_PRC|PORTSC_PLC|PORTSC_CEC));

        if (val & PORTSC_LWS) {

            /* overwrite PLS only when LWS=1 */

            uint32_t old_pls = get_field(port->portsc, PORTSC_PLS);

            uint32_t new_pls = get_field(val, PORTSC_PLS);

            switch (new_pls) {

            case PLS_U0:

                if (old_pls != PLS_U0) {

                    set_field(&portsc, new_pls, PORTSC_PLS);

                    trace_usb_xhci_port_link(port->portnr, new_pls);

                    notify = PORTSC_PLC;

                }

                break;

            case PLS_U3:

                if (old_pls < PLS_U3) {

                    set_field(&portsc, new_pls, PORTSC_PLS);

                    trace_usb_xhci_port_link(port->portnr, new_pls);

                }

                break;

            case PLS_RESUME:

                /* windows does this for some reason, don't spam stderr */

                break;

            default:

                fprintf(stderr, "%s: ignore pls write (old %d, new %d)\n",

                        __func__, old_pls, new_pls);

                break;

            }

        }

        /* read/write bits */

        portsc &= ~(PORTSC_PP|PORTSC_WCE|PORTSC_WDE|PORTSC_WOE);

        portsc |= (val & (PORTSC_PP|PORTSC_WCE|PORTSC_WDE|PORTSC_WOE));

        port->portsc = portsc;

        if (notify) {

            xhci_port_notify(port, notify);

        }

        break;

    case 0x04: /* PORTPMSC */

    case 0x08: /* PORTLI */

    default:

        trace_usb_xhci_unimplemented("port write", reg);

    }

}

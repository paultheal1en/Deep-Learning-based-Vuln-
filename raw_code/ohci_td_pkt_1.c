static void ohci_td_pkt(const char *msg, const uint8_t *buf, size_t len)

{

    bool print16 = !!trace_event_get_state(TRACE_USB_OHCI_TD_PKT_SHORT);

    bool printall = !!trace_event_get_state(TRACE_USB_OHCI_TD_PKT_FULL);

    const int width = 16;

    int i;

    char tmp[3 * width + 1];

    char *p = tmp;



    if (!printall && !print16) {

        return;

    }



    for (i = 0; ; i++) {

        if (i && (!(i % width) || (i == len))) {

            if (!printall) {

                trace_usb_ohci_td_pkt_short(msg, tmp);

                break;

            }

            trace_usb_ohci_td_pkt_full(msg, tmp);

            p = tmp;

            *p = 0;

        }

        if (i == len) {

            break;

        }



        p += sprintf(p, " %.2x", buf[i]);

    }

}

static int ehci_state_executing(EHCIQueue *q, int async)

{

    int again = 0;

    int reload, nakcnt;



    ehci_execute_complete(q);

    if (q->usb_status == USB_RET_ASYNC) {

        goto out;

    }

    if (q->usb_status == USB_RET_PROCERR) {

        again = -1;

        goto out;

    }



    // 4.10.3

    if (!async) {

        int transactCtr = get_field(q->qh.epcap, QH_EPCAP_MULT);

        transactCtr--;

        set_field(&q->qh.epcap, transactCtr, QH_EPCAP_MULT);

        // 4.10.3, bottom of page 82, should exit this state when transaction

        // counter decrements to 0

    }



    reload = get_field(q->qh.epchar, QH_EPCHAR_RL);

    if (reload) {

        nakcnt = get_field(q->qh.altnext_qtd, QH_ALTNEXT_NAKCNT);

        if (q->usb_status == USB_RET_NAK) {

            if (nakcnt) {

                nakcnt--;

            }

        } else {

            nakcnt = reload;

        }

        set_field(&q->qh.altnext_qtd, nakcnt, QH_ALTNEXT_NAKCNT);

    }



    /* 4.10.5 */

    if ((q->usb_status == USB_RET_NAK) || (q->qh.token & QTD_TOKEN_ACTIVE)) {

        ehci_set_state(q->ehci, async, EST_HORIZONTALQH);

    } else {

        ehci_set_state(q->ehci, async, EST_WRITEBACK);

    }



    again = 1;



out:

    ehci_flush_qh(q);

    return again;

}

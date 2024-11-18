static void xhci_events_update(XHCIState *xhci, int v)

{

    XHCIInterrupter *intr = &xhci->intr[v];

    dma_addr_t erdp;

    unsigned int dp_idx;

    bool do_irq = 0;



    if (xhci->usbsts & USBSTS_HCH) {

        return;

    }



    erdp = xhci_addr64(intr->erdp_low, intr->erdp_high);

    if (erdp < intr->er_start ||

        erdp >= (intr->er_start + TRB_SIZE*intr->er_size)) {

        DPRINTF("xhci: ERDP out of bounds: "DMA_ADDR_FMT"\n", erdp);

        DPRINTF("xhci: ER[%d] at "DMA_ADDR_FMT" len %d\n",

                v, intr->er_start, intr->er_size);

        xhci_die(xhci);

        return;

    }

    dp_idx = (erdp - intr->er_start) / TRB_SIZE;

    assert(dp_idx < intr->er_size);



    /* NEC didn't read section 4.9.4 of the spec (v1.0 p139 top Note) and thus

     * deadlocks when the ER is full. Hack it by holding off events until

     * the driver decides to free at least half of the ring */

    if (intr->er_full) {

        int er_free = dp_idx - intr->er_ep_idx;

        if (er_free <= 0) {

            er_free += intr->er_size;

        }

        if (er_free < (intr->er_size/2)) {

            DPRINTF("xhci_events_update(): event ring still "

                    "more than half full (hack)\n");

            return;

        }

    }



    while (intr->ev_buffer_put != intr->ev_buffer_get) {

        assert(intr->er_full);

        if (((intr->er_ep_idx+1) % intr->er_size) == dp_idx) {

            DPRINTF("xhci_events_update(): event ring full again\n");

#ifndef ER_FULL_HACK

            XHCIEvent full = {ER_HOST_CONTROLLER, CC_EVENT_RING_FULL_ERROR};

            xhci_write_event(xhci, &full, v);

#endif

            do_irq = 1;

            break;

        }

        XHCIEvent *event = &intr->ev_buffer[intr->ev_buffer_get];

        xhci_write_event(xhci, event, v);

        intr->ev_buffer_get++;

        do_irq = 1;

        if (intr->ev_buffer_get == EV_QUEUE) {

            intr->ev_buffer_get = 0;

        }

    }



    if (do_irq) {

        xhci_intr_raise(xhci, v);

    }



    if (intr->er_full && intr->ev_buffer_put == intr->ev_buffer_get) {

        DPRINTF("xhci_events_update(): event ring no longer full\n");

        intr->er_full = 0;

    }

}

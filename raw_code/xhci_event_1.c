static void xhci_event(XHCIState *xhci, XHCIEvent *event, int v)

{

    XHCIInterrupter *intr;

    dma_addr_t erdp;

    unsigned int dp_idx;



    if (v >= xhci->numintrs) {

        DPRINTF("intr nr out of range (%d >= %d)\n", v, xhci->numintrs);

        return;

    }

    intr = &xhci->intr[v];



    if (intr->er_full) {

        DPRINTF("xhci_event(): ER full, queueing\n");

        if (((intr->ev_buffer_put+1) % EV_QUEUE) == intr->ev_buffer_get) {

            DPRINTF("xhci: event queue full, dropping event!\n");

            return;

        }

        intr->ev_buffer[intr->ev_buffer_put++] = *event;

        if (intr->ev_buffer_put == EV_QUEUE) {

            intr->ev_buffer_put = 0;

        }

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



    if ((intr->er_ep_idx+1) % intr->er_size == dp_idx) {

        DPRINTF("xhci_event(): ER full, queueing\n");

#ifndef ER_FULL_HACK

        XHCIEvent full = {ER_HOST_CONTROLLER, CC_EVENT_RING_FULL_ERROR};

        xhci_write_event(xhci, &full);

#endif

        intr->er_full = 1;

        if (((intr->ev_buffer_put+1) % EV_QUEUE) == intr->ev_buffer_get) {

            DPRINTF("xhci: event queue full, dropping event!\n");

            return;

        }

        intr->ev_buffer[intr->ev_buffer_put++] = *event;

        if (intr->ev_buffer_put == EV_QUEUE) {

            intr->ev_buffer_put = 0;

        }

    } else {

        xhci_write_event(xhci, event, v);

    }



    xhci_intr_raise(xhci, v);

}

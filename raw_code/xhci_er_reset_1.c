static void xhci_er_reset(XHCIState *xhci, int v)

{

    XHCIInterrupter *intr = &xhci->intr[v];

    XHCIEvRingSeg seg;



    if (intr->erstsz == 0) {

        /* disabled */

        intr->er_start = 0;

        intr->er_size = 0;

        return;

    }

    /* cache the (sole) event ring segment location */

    if (intr->erstsz != 1) {

        DPRINTF("xhci: invalid value for ERSTSZ: %d\n", intr->erstsz);

        xhci_die(xhci);

        return;

    }

    dma_addr_t erstba = xhci_addr64(intr->erstba_low, intr->erstba_high);

    pci_dma_read(PCI_DEVICE(xhci), erstba, &seg, sizeof(seg));

    le32_to_cpus(&seg.addr_low);

    le32_to_cpus(&seg.addr_high);

    le32_to_cpus(&seg.size);

    if (seg.size < 16 || seg.size > 4096) {

        DPRINTF("xhci: invalid value for segment size: %d\n", seg.size);

        xhci_die(xhci);

        return;

    }

    intr->er_start = xhci_addr64(seg.addr_low, seg.addr_high);

    intr->er_size = seg.size;



    intr->er_ep_idx = 0;

    intr->er_pcs = 1;

    intr->er_full = 0;



    DPRINTF("xhci: event ring[%d]:" DMA_ADDR_FMT " [%d]\n",

            v, intr->er_start, intr->er_size);

}

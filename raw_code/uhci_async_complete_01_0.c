static void uhci_async_complete(USBPort *port, USBPacket *packet)

{

    UHCIAsync *async = container_of(packet, UHCIAsync, packet);

    UHCIState *s = async->queue->uhci;



    if (async->isoc) {

        UHCI_TD td;

        uint32_t link = async->td;

        uint32_t int_mask = 0, val;



        pci_dma_read(&s->dev, link & ~0xf, &td, sizeof(td));

        le32_to_cpus(&td.link);

        le32_to_cpus(&td.ctrl);

        le32_to_cpus(&td.token);

        le32_to_cpus(&td.buffer);



        uhci_async_unlink(async);

        uhci_complete_td(s, &td, async, &int_mask);

        s->pending_int_mask |= int_mask;



        /* update the status bits of the TD */

        val = cpu_to_le32(td.ctrl);

        pci_dma_write(&s->dev, (link & ~0xf) + 4, &val, sizeof(val));

        uhci_async_free(async);

    } else {

        async->done = 1;

        uhci_process_frame(s);

    }

}

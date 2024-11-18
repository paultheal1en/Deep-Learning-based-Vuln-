static void esp_command_complete(SCSIBus *bus, int reason, uint32_t tag,

                                 uint32_t arg)

{

    ESPState *s = DO_UPCAST(ESPState, busdev.qdev, bus->qbus.parent);



    if (reason == SCSI_REASON_DONE) {

        DPRINTF("SCSI Command complete\n");

        if (s->ti_size != 0)

            DPRINTF("SCSI command completed unexpectedly\n");

        s->ti_size = 0;

        s->dma_left = 0;

        s->async_len = 0;

        if (arg)

            DPRINTF("Command failed\n");

        s->sense = arg;

        s->rregs[ESP_RSTAT] = STAT_ST;

        esp_dma_done(s);

        s->current_dev = NULL;

    } else {

        DPRINTF("transfer %d/%d\n", s->dma_left, s->ti_size);

        s->async_len = arg;

        s->async_buf = s->current_dev->info->get_buf(s->current_dev, 0);

        if (s->dma_left) {

            esp_do_dma(s);

        } else if (s->dma_counter != 0 && s->ti_size <= 0) {

            /* If this was the last part of a DMA transfer then the

               completion interrupt is deferred to here.  */

            esp_dma_done(s);

        }

    }

}

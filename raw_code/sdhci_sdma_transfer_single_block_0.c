static void sdhci_sdma_transfer_single_block(SDHCIState *s)

{

    int n;

    uint32_t datacnt = s->blksize & 0x0fff;



    if (s->trnmod & SDHC_TRNS_READ) {

        for (n = 0; n < datacnt; n++) {

            s->fifo_buffer[n] = sdbus_read_data(&s->sdbus);

        }

        dma_memory_write(&address_space_memory, s->sdmasysad, s->fifo_buffer,

                         datacnt);

    } else {

        dma_memory_read(&address_space_memory, s->sdmasysad, s->fifo_buffer,

                        datacnt);

        for (n = 0; n < datacnt; n++) {

            sdbus_write_data(&s->sdbus, s->fifo_buffer[n]);

        }

    }



    if (s->trnmod & SDHC_TRNS_BLK_CNT_EN) {

        s->blkcnt--;

    }



    sdhci_end_transfer(s);

}

void bmdma_cmd_writeb(BMDMAState *bm, uint32_t val)

{

#ifdef DEBUG_IDE

    printf("%s: 0x%08x\n", __func__, val);

#endif



    /* Ignore writes to SSBM if it keeps the old value */

    if ((val & BM_CMD_START) != (bm->cmd & BM_CMD_START)) {

        if (!(val & BM_CMD_START)) {

            /*

             * We can't cancel Scatter Gather DMA in the middle of the

             * operation or a partial (not full) DMA transfer would reach

             * the storage so we wait for completion instead (we beahve

             * like if the DMA was completed by the time the guest trying

             * to cancel dma with bmdma_cmd_writeb with BM_CMD_START not

             * set).

             *

             * In the future we'll be able to safely cancel the I/O if the

             * whole DMA operation will be submitted to disk with a single

             * aio operation with preadv/pwritev.

             */

            if (bm->bus->dma->aiocb) {

                qemu_aio_flush();

                assert(bm->bus->dma->aiocb == NULL);

                assert((bm->status & BM_STATUS_DMAING) == 0);

            }

        } else {

            bm->cur_addr = bm->addr;

            if (!(bm->status & BM_STATUS_DMAING)) {

                bm->status |= BM_STATUS_DMAING;

                /* start dma transfer if possible */

                if (bm->dma_cb)

                    bm->dma_cb(bmdma_active_if(bm), 0);

            }

        }

    }



    bm->cmd = val & 0x09;

}

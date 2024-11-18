static void pmac_ide_transfer_cb(void *opaque, int ret)

{

    DBDMA_io *io = opaque;

    MACIOIDEState *m = io->opaque;

    IDEState *s = idebus_active_if(&m->bus);

    int64_t sector_num;

    int nsector, remainder;

    int64_t offset;



    MACIO_DPRINTF("pmac_ide_transfer_cb\n");



    if (ret < 0) {

        MACIO_DPRINTF("DMA error\n");

        m->aiocb = NULL;

        ide_dma_error(s);

        io->remainder_len = 0;

        goto done;

    }



    if (!m->dma_active) {

        MACIO_DPRINTF("waiting for data (%#x - %#x - %x)\n",

                      s->nsector, io->len, s->status);

        /* data not ready yet, wait for the channel to get restarted */

        io->processing = false;

        return;

    }



    if (s->io_buffer_size <= 0) {

        MACIO_DPRINTF("end of transfer\n");

        s->status = READY_STAT | SEEK_STAT;

        ide_set_irq(s->bus);

        m->dma_active = false;

        goto done;

    }



    if (io->len == 0) {

        MACIO_DPRINTF("End of DMA transfer\n");

        goto done;

    }



    /* Calculate number of sectors */

    sector_num = ide_get_sector(s) + (s->io_buffer_index >> 9);

    offset = (ide_get_sector(s) << 9) + s->io_buffer_index;

    nsector = (io->len + 0x1ff) >> 9;

    remainder = io->len & 0x1ff;



    s->nsector -= nsector;



    MACIO_DPRINTF("nsector: %d   remainder: %x\n", nsector, remainder);

    MACIO_DPRINTF("sector: %"PRIx64"   %x\n", sector_num, nsector);



    switch (s->dma_cmd) {

    case IDE_DMA_READ:

        pmac_dma_read(s->blk, offset, io->len, pmac_ide_transfer_cb, io);

        break;

    case IDE_DMA_WRITE:

        pmac_dma_write(s->blk, sector_num, nsector, pmac_ide_transfer_cb, io);

        break;

    case IDE_DMA_TRIM:

        MACIO_DPRINTF("TRIM command issued!");

        break;

    }



    return;



done:

    if (s->dma_cmd == IDE_DMA_READ || s->dma_cmd == IDE_DMA_WRITE) {

        block_acct_done(blk_get_stats(s->blk), &s->acct);

    }

    io->dma_end(opaque);

}

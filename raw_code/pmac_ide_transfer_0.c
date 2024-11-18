static void pmac_ide_transfer(DBDMA_io *io)

{

    MACIOIDEState *m = io->opaque;

    IDEState *s = idebus_active_if(&m->bus);



    MACIO_DPRINTF("\n");



    s->io_buffer_size = 0;

    if (s->drive_kind == IDE_CD) {



        /* Handle non-block ATAPI DMA transfers */

        if (s->lba == -1) {

            s->io_buffer_size = MIN(io->len, s->packet_transfer_size);

            block_acct_start(bdrv_get_stats(s->bs), &s->acct, s->io_buffer_size,

                             BLOCK_ACCT_READ);

            MACIO_DPRINTF("non-block ATAPI DMA transfer size: %d\n",

                          s->io_buffer_size);



            /* Copy ATAPI buffer directly to RAM and finish */

            cpu_physical_memory_write(io->addr, s->io_buffer,

                                      s->io_buffer_size);

            ide_atapi_cmd_ok(s);

            m->dma_active = false;



            MACIO_DPRINTF("end of non-block ATAPI DMA transfer\n");

            block_acct_done(bdrv_get_stats(s->bs), &s->acct);

            io->dma_end(io);

            return;

        }



        block_acct_start(bdrv_get_stats(s->bs), &s->acct, io->len,

                         BLOCK_ACCT_READ);

        pmac_ide_atapi_transfer_cb(io, 0);

        return;

    }



    switch (s->dma_cmd) {

    case IDE_DMA_READ:

        block_acct_start(bdrv_get_stats(s->bs), &s->acct, io->len,

                         BLOCK_ACCT_READ);

        break;

    case IDE_DMA_WRITE:

        block_acct_start(bdrv_get_stats(s->bs), &s->acct, io->len,

                         BLOCK_ACCT_WRITE);

        break;

    default:

        break;

    }



    io->requests++;

    pmac_ide_transfer_cb(io, 0);

}

static void ide_atapi_cmd_read_dma(IDEState *s, int lba, int nb_sectors,

                                   int sector_size)

{

    s->lba = lba;

    s->packet_transfer_size = nb_sectors * sector_size;

    s->io_buffer_index = 0;

    s->io_buffer_size = 0;

    s->cd_sector_size = sector_size;



    block_acct_start(bdrv_get_stats(s->bs), &s->acct, s->packet_transfer_size,

                     BLOCK_ACCT_READ);



    /* XXX: check if BUSY_STAT should be set */

    s->status = READY_STAT | SEEK_STAT | DRQ_STAT | BUSY_STAT;

    ide_start_dma(s, ide_atapi_cmd_read_dma_cb);

}

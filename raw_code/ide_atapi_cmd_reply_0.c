static void ide_atapi_cmd_reply(IDEState *s, int size, int max_size)

{

    if (size > max_size)

        size = max_size;

    s->lba = -1; /* no sector read */

    s->packet_transfer_size = size;

    s->io_buffer_size = size;    /* dma: send the reply data as one chunk */

    s->elementary_transfer_size = 0;

    s->io_buffer_index = 0;



    if (s->atapi_dma) {

        block_acct_start(bdrv_get_stats(s->bs), &s->acct, size,

                         BLOCK_ACCT_READ);

        s->status = READY_STAT | SEEK_STAT | DRQ_STAT;

        ide_start_dma(s, ide_atapi_cmd_read_dma_cb);

    } else {

        s->status = READY_STAT | SEEK_STAT;

        ide_atapi_cmd_reply_end(s);

    }

}

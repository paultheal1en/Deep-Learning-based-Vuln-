static void ide_dma_cb(void *opaque, int ret)

{

    IDEState *s = opaque;

    int n;

    int64_t sector_num;

    uint64_t offset;

    bool stay_active = false;



    if (ret == -ECANCELED) {

        return;

    }

    if (ret < 0) {

        if (ide_handle_rw_error(s, -ret, ide_dma_cmd_to_retry(s->dma_cmd))) {

            s->bus->dma->aiocb = NULL;

            dma_buf_commit(s, 0);

            return;

        }

    }



    n = s->io_buffer_size >> 9;

    if (n > s->nsector) {

        /* The PRDs were longer than needed for this request. Shorten them so

         * we don't get a negative remainder. The Active bit must remain set

         * after the request completes. */

        n = s->nsector;

        stay_active = true;

    }



    sector_num = ide_get_sector(s);

    if (n > 0) {

        assert(n * 512 == s->sg.size);

        dma_buf_commit(s, s->sg.size);

        sector_num += n;

        ide_set_sector(s, sector_num);

        s->nsector -= n;

    }



    /* end of transfer ? */

    if (s->nsector == 0) {

        s->status = READY_STAT | SEEK_STAT;

        ide_set_irq(s->bus);

        goto eot;

    }



    /* launch next transfer */

    n = s->nsector;

    s->io_buffer_index = 0;

    s->io_buffer_size = n * 512;

    if (s->bus->dma->ops->prepare_buf(s->bus->dma, s->io_buffer_size) < 512) {

        /* The PRDs were too short. Reset the Active bit, but don't raise an

         * interrupt. */

        s->status = READY_STAT | SEEK_STAT;

        dma_buf_commit(s, 0);

        goto eot;

    }



    trace_ide_dma_cb(s, sector_num, n, IDE_DMA_CMD_str(s->dma_cmd));



    if ((s->dma_cmd == IDE_DMA_READ || s->dma_cmd == IDE_DMA_WRITE) &&

        !ide_sect_range_ok(s, sector_num, n)) {

        ide_dma_error(s);

        block_acct_invalid(blk_get_stats(s->blk), s->acct.type);

        return;

    }



    offset = sector_num << BDRV_SECTOR_BITS;

    switch (s->dma_cmd) {

    case IDE_DMA_READ:

        s->bus->dma->aiocb = dma_blk_read(s->blk, &s->sg, offset,

                                          BDRV_SECTOR_SIZE, ide_dma_cb, s);

        break;

    case IDE_DMA_WRITE:

        s->bus->dma->aiocb = dma_blk_write(s->blk, &s->sg, offset,

                                           BDRV_SECTOR_SIZE, ide_dma_cb, s);

        break;

    case IDE_DMA_TRIM:

        s->bus->dma->aiocb = dma_blk_io(blk_get_aio_context(s->blk),

                                        &s->sg, offset, BDRV_SECTOR_SIZE,

                                        ide_issue_trim, s->blk, ide_dma_cb, s,

                                        DMA_DIRECTION_TO_DEVICE);

        break;

    default:

        abort();

    }

    return;



eot:

    if (s->dma_cmd == IDE_DMA_READ || s->dma_cmd == IDE_DMA_WRITE) {

        block_acct_done(blk_get_stats(s->blk), &s->acct);

    }

    ide_set_inactive(s, stay_active);

}

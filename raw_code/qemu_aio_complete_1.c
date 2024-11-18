static void qemu_aio_complete(void *opaque, int ret)

{

    struct ioreq *ioreq = opaque;



    if (ret != 0) {

        xen_be_printf(&ioreq->blkdev->xendev, 0, "%s I/O error\n",

                      ioreq->req.operation == BLKIF_OP_READ ? "read" : "write");

        ioreq->aio_errors++;

    }



    ioreq->aio_inflight--;

    if (ioreq->presync) {

        ioreq->presync = 0;

        ioreq_runio_qemu_aio(ioreq);

        return;

    }

    if (ioreq->aio_inflight > 0) {

        return;

    }

    if (ioreq->postsync) {

        ioreq->postsync = 0;

        ioreq->aio_inflight++;

        blk_aio_flush(ioreq->blkdev->blk, qemu_aio_complete, ioreq);

        return;

    }



    ioreq->status = ioreq->aio_errors ? BLKIF_RSP_ERROR : BLKIF_RSP_OKAY;

    ioreq_unmap(ioreq);

    ioreq_finish(ioreq);

    switch (ioreq->req.operation) {

    case BLKIF_OP_WRITE:

    case BLKIF_OP_FLUSH_DISKCACHE:

        if (!ioreq->req.nr_segments) {

            break;

        }

    case BLKIF_OP_READ:

        block_acct_done(blk_get_stats(ioreq->blkdev->blk), &ioreq->acct);

        break;

    case BLKIF_OP_DISCARD:

    default:

        break;

    }

    qemu_bh_schedule(ioreq->blkdev->bh);

}

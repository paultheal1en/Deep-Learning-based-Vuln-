static void ioreq_release(struct ioreq *ioreq, bool finish)

{

    struct XenBlkDev *blkdev = ioreq->blkdev;



    QLIST_REMOVE(ioreq, list);

    memset(ioreq, 0, sizeof(*ioreq));

    ioreq->blkdev = blkdev;

    QLIST_INSERT_HEAD(&blkdev->freelist, ioreq, list);

    if (finish) {

        blkdev->requests_finished--;

    } else {

        blkdev->requests_inflight--;

    }

}

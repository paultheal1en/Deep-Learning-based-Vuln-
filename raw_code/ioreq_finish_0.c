static void ioreq_finish(struct ioreq *ioreq)

{

    struct XenBlkDev *blkdev = ioreq->blkdev;



    LIST_REMOVE(ioreq, list);

    LIST_INSERT_HEAD(&blkdev->finished, ioreq, list);

    blkdev->requests_inflight--;

    blkdev->requests_finished++;

}

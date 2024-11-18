static struct ioreq *ioreq_start(struct XenBlkDev *blkdev)

{

    struct ioreq *ioreq = NULL;



    if (LIST_EMPTY(&blkdev->freelist)) {

	if (blkdev->requests_total >= max_requests)

	    goto out;

	/* allocate new struct */

	ioreq = qemu_mallocz(sizeof(*ioreq));

	ioreq->blkdev = blkdev;

	blkdev->requests_total++;

        qemu_iovec_init(&ioreq->v, BLKIF_MAX_SEGMENTS_PER_REQUEST);

    } else {

	/* get one from freelist */

	ioreq = LIST_FIRST(&blkdev->freelist);

	LIST_REMOVE(ioreq, list);

        qemu_iovec_reset(&ioreq->v);

    }

    LIST_INSERT_HEAD(&blkdev->inflight, ioreq, list);

    blkdev->requests_inflight++;



out:

    return ioreq;

}

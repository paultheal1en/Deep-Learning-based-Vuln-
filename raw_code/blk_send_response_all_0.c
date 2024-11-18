static void blk_send_response_all(struct XenBlkDev *blkdev)

{

    struct ioreq *ioreq;

    int send_notify = 0;



    while (!LIST_EMPTY(&blkdev->finished)) {

        ioreq = LIST_FIRST(&blkdev->finished);

	send_notify += blk_send_response_one(ioreq);

	ioreq_release(ioreq);

    }

    if (send_notify)

	xen_be_send_notify(&blkdev->xendev);

}

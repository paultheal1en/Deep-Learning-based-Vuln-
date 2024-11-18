static void blk_alloc(struct XenDevice *xendev)

{

    struct XenBlkDev *blkdev = container_of(xendev, struct XenBlkDev, xendev);



    LIST_INIT(&blkdev->inflight);

    LIST_INIT(&blkdev->finished);

    LIST_INIT(&blkdev->freelist);

    blkdev->bh = qemu_bh_new(blk_bh, blkdev);

    if (xen_mode != XEN_EMULATE)

        batch_maps = 1;

}

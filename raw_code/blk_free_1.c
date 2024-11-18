static int blk_free(struct XenDevice *xendev)

{

    struct XenBlkDev *blkdev = container_of(xendev, struct XenBlkDev, xendev);

    struct ioreq *ioreq;



    if (blkdev->blk || blkdev->sring) {

        blk_disconnect(xendev);

    }



    /* Free persistent grants */

    if (blkdev->feature_persistent) {

        g_tree_destroy(blkdev->persistent_gnts);

    }



    while (!QLIST_EMPTY(&blkdev->freelist)) {

        ioreq = QLIST_FIRST(&blkdev->freelist);

        QLIST_REMOVE(ioreq, list);

        qemu_iovec_destroy(&ioreq->v);

        g_free(ioreq);

    }



    g_free(blkdev->params);

    g_free(blkdev->mode);

    g_free(blkdev->type);

    g_free(blkdev->dev);

    g_free(blkdev->devtype);

    qemu_bh_delete(blkdev->bh);

    return 0;

}

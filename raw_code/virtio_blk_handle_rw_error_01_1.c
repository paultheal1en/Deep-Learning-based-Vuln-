static int virtio_blk_handle_rw_error(VirtIOBlockReq *req, int error,

    int is_read)

{

    BlockInterfaceErrorAction action =

        drive_get_on_error(req->dev->bs, is_read);

    VirtIOBlock *s = req->dev;



    if (action == BLOCK_ERR_IGNORE) {

        bdrv_mon_event(req->dev->bs, BDRV_ACTION_IGNORE, is_read);

        return 0;

    }



    if ((error == ENOSPC && action == BLOCK_ERR_STOP_ENOSPC)

            || action == BLOCK_ERR_STOP_ANY) {

        req->next = s->rq;

        s->rq = req;

        bdrv_mon_event(req->dev->bs, BDRV_ACTION_STOP, is_read);

        vm_stop(0);

    } else {

        virtio_blk_req_complete(req, VIRTIO_BLK_S_IOERR);

        bdrv_mon_event(req->dev->bs, BDRV_ACTION_REPORT, is_read);

    }



    return 1;

}

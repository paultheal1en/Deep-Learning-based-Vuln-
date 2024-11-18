static int virtio_blk_handle_scsi_req(VirtIOBlockReq *req)

{

    int status = VIRTIO_BLK_S_OK;

    struct virtio_scsi_inhdr *scsi = NULL;

    VirtIODevice *vdev = VIRTIO_DEVICE(req->dev);

    VirtQueueElement *elem = &req->elem;

    VirtIOBlock *blk = req->dev;



#ifdef __linux__

    int i;

    VirtIOBlockIoctlReq *ioctl_req;

#endif



    /*

     * We require at least one output segment each for the virtio_blk_outhdr

     * and the SCSI command block.

     *

     * We also at least require the virtio_blk_inhdr, the virtio_scsi_inhdr

     * and the sense buffer pointer in the input segments.

     */

    if (elem->out_num < 2 || elem->in_num < 3) {

        status = VIRTIO_BLK_S_IOERR;

        goto fail;

    }



    /*

     * The scsi inhdr is placed in the second-to-last input segment, just

     * before the regular inhdr.

     */

    scsi = (void *)elem->in_sg[elem->in_num - 2].iov_base;



    if (!blk->conf.scsi) {

        status = VIRTIO_BLK_S_UNSUPP;

        goto fail;

    }



    /*

     * No support for bidirection commands yet.

     */

    if (elem->out_num > 2 && elem->in_num > 3) {

        status = VIRTIO_BLK_S_UNSUPP;

        goto fail;

    }



#ifdef __linux__

    ioctl_req = g_new0(VirtIOBlockIoctlReq, 1);

    ioctl_req->req = req;

    ioctl_req->hdr.interface_id = 'S';

    ioctl_req->hdr.cmd_len = elem->out_sg[1].iov_len;

    ioctl_req->hdr.cmdp = elem->out_sg[1].iov_base;

    ioctl_req->hdr.dxfer_len = 0;



    if (elem->out_num > 2) {

        /*

         * If there are more than the minimally required 2 output segments

         * there is write payload starting from the third iovec.

         */

        ioctl_req->hdr.dxfer_direction = SG_DXFER_TO_DEV;

        ioctl_req->hdr.iovec_count = elem->out_num - 2;



        for (i = 0; i < ioctl_req->hdr.iovec_count; i++) {

            ioctl_req->hdr.dxfer_len += elem->out_sg[i + 2].iov_len;

        }



        ioctl_req->hdr.dxferp = elem->out_sg + 2;



    } else if (elem->in_num > 3) {

        /*

         * If we have more than 3 input segments the guest wants to actually

         * read data.

         */

        ioctl_req->hdr.dxfer_direction = SG_DXFER_FROM_DEV;

        ioctl_req->hdr.iovec_count = elem->in_num - 3;

        for (i = 0; i < ioctl_req->hdr.iovec_count; i++) {

            ioctl_req->hdr.dxfer_len += elem->in_sg[i].iov_len;

        }



        ioctl_req->hdr.dxferp = elem->in_sg;

    } else {

        /*

         * Some SCSI commands don't actually transfer any data.

         */

        ioctl_req->hdr.dxfer_direction = SG_DXFER_NONE;

    }



    ioctl_req->hdr.sbp = elem->in_sg[elem->in_num - 3].iov_base;

    ioctl_req->hdr.mx_sb_len = elem->in_sg[elem->in_num - 3].iov_len;



    blk_aio_ioctl(blk->blk, SG_IO, &ioctl_req->hdr,

                  virtio_blk_ioctl_complete, ioctl_req);

    return -EINPROGRESS;

#else

    abort();

#endif



fail:

    /* Just put anything nonzero so that the ioctl fails in the guest.  */

    if (scsi) {

        virtio_stl_p(vdev, &scsi->errors, 255);

    }

    return status;

}

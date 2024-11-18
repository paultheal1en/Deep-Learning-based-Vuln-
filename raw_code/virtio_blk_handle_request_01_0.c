static void virtio_blk_handle_request(VirtIOBlockReq *req,

    MultiReqBuffer *mrb)

{

    uint32_t type;

    struct iovec *iov = req->elem->out_sg;

    unsigned out_num = req->elem->out_num;



    if (req->elem->out_num < 1 || req->elem->in_num < 1) {

        error_report("virtio-blk missing headers");

        exit(1);

    }



    if (req->elem->out_sg[0].iov_len < sizeof(req->out) ||

        req->elem->in_sg[req->elem->in_num - 1].iov_len < sizeof(*req->in)) {

        error_report("virtio-blk header not in correct element");

        exit(1);

    }



    if (unlikely(iov_to_buf(iov, out_num, 0, &req->out,

                            sizeof(req->out)) != sizeof(req->out))) {

        error_report("virtio-blk request outhdr too short");

        exit(1);

    }

    iov_discard_front(&iov, &out_num, sizeof(req->out));

    req->in = (void *)req->elem->in_sg[req->elem->in_num - 1].iov_base;



    type = ldl_p(&req->out.type);



    if (type & VIRTIO_BLK_T_FLUSH) {

        virtio_blk_handle_flush(req, mrb);

    } else if (type & VIRTIO_BLK_T_SCSI_CMD) {

        virtio_blk_handle_scsi(req);

    } else if (type & VIRTIO_BLK_T_GET_ID) {

        VirtIOBlock *s = req->dev;



        /*

         * NB: per existing s/n string convention the string is

         * terminated by '\0' only when shorter than buffer.

         */

        strncpy(req->elem->in_sg[0].iov_base,

                s->blk.serial ? s->blk.serial : "",

                MIN(req->elem->in_sg[0].iov_len, VIRTIO_BLK_ID_BYTES));

        virtio_blk_req_complete(req, VIRTIO_BLK_S_OK);

        virtio_blk_free_request(req);

    } else if (type & VIRTIO_BLK_T_OUT) {

        qemu_iovec_init_external(&req->qiov, &req->elem->out_sg[1],

                                 req->elem->out_num - 1);

        virtio_blk_handle_write(req, mrb);

    } else if (type == VIRTIO_BLK_T_IN || type == VIRTIO_BLK_T_BARRIER) {

        /* VIRTIO_BLK_T_IN is 0, so we can't just & it. */

        qemu_iovec_init_external(&req->qiov, &req->elem->in_sg[0],

                                 req->elem->in_num - 1);

        virtio_blk_handle_read(req);

    } else {

        virtio_blk_req_complete(req, VIRTIO_BLK_S_UNSUPP);

        virtio_blk_free_request(req);

    }

}

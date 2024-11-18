void virtio_blk_handle_request(VirtIOBlockReq *req, MultiReqBuffer *mrb)

{

    uint32_t type;

    struct iovec *in_iov = req->elem.in_sg;

    struct iovec *iov = req->elem.out_sg;

    unsigned in_num = req->elem.in_num;

    unsigned out_num = req->elem.out_num;



    if (req->elem.out_num < 1 || req->elem.in_num < 1) {

        error_report("virtio-blk missing headers");

        exit(1);

    }



    if (unlikely(iov_to_buf(iov, out_num, 0, &req->out,

                            sizeof(req->out)) != sizeof(req->out))) {

        error_report("virtio-blk request outhdr too short");

        exit(1);

    }



    iov_discard_front(&iov, &out_num, sizeof(req->out));



    if (in_num < 1 ||

        in_iov[in_num - 1].iov_len < sizeof(struct virtio_blk_inhdr)) {

        error_report("virtio-blk request inhdr too short");

        exit(1);

    }



    /* We always touch the last byte, so just see how big in_iov is.  */

    req->in_len = iov_size(in_iov, in_num);

    req->in = (void *)in_iov[in_num - 1].iov_base

              + in_iov[in_num - 1].iov_len

              - sizeof(struct virtio_blk_inhdr);

    iov_discard_back(in_iov, &in_num, sizeof(struct virtio_blk_inhdr));



    type = virtio_ldl_p(VIRTIO_DEVICE(req->dev), &req->out.type);



    /* VIRTIO_BLK_T_OUT defines the command direction. VIRTIO_BLK_T_BARRIER

     * is an optional flag. Although a guest should not send this flag if

     * not negotiated we ignored it in the past. So keep ignoring it. */

    switch (type & ~(VIRTIO_BLK_T_OUT | VIRTIO_BLK_T_BARRIER)) {

    case VIRTIO_BLK_T_IN:

    {

        bool is_write = type & VIRTIO_BLK_T_OUT;

        req->sector_num = virtio_ldq_p(VIRTIO_DEVICE(req->dev),

                                       &req->out.sector);



        if (is_write) {

            qemu_iovec_init_external(&req->qiov, iov, out_num);

            trace_virtio_blk_handle_write(req, req->sector_num,

                                          req->qiov.size / BDRV_SECTOR_SIZE);

        } else {

            qemu_iovec_init_external(&req->qiov, in_iov, in_num);

            trace_virtio_blk_handle_read(req, req->sector_num,

                                         req->qiov.size / BDRV_SECTOR_SIZE);

        }



        if (!virtio_blk_sect_range_ok(req->dev, req->sector_num,

                                      req->qiov.size)) {

            virtio_blk_req_complete(req, VIRTIO_BLK_S_IOERR);

            virtio_blk_free_request(req);

            return;

        }



        block_acct_start(blk_get_stats(req->dev->blk),

                         &req->acct, req->qiov.size,

                         is_write ? BLOCK_ACCT_WRITE : BLOCK_ACCT_READ);



        /* merge would exceed maximum number of requests or IO direction

         * changes */

        if (mrb->num_reqs > 0 && (mrb->num_reqs == VIRTIO_BLK_MAX_MERGE_REQS ||

                                  is_write != mrb->is_write ||

                                  !req->dev->conf.request_merging)) {

            virtio_blk_submit_multireq(req->dev->blk, mrb);

        }



        assert(mrb->num_reqs < VIRTIO_BLK_MAX_MERGE_REQS);

        mrb->reqs[mrb->num_reqs++] = req;

        mrb->is_write = is_write;

        break;

    }

    case VIRTIO_BLK_T_FLUSH:

        virtio_blk_handle_flush(req, mrb);

        break;

    case VIRTIO_BLK_T_SCSI_CMD:

        virtio_blk_handle_scsi(req);

        break;

    case VIRTIO_BLK_T_GET_ID:

    {

        VirtIOBlock *s = req->dev;



        /*

         * NB: per existing s/n string convention the string is

         * terminated by '\0' only when shorter than buffer.

         */

        const char *serial = s->conf.serial ? s->conf.serial : "";

        size_t size = MIN(strlen(serial) + 1,

                          MIN(iov_size(in_iov, in_num),

                              VIRTIO_BLK_ID_BYTES));

        iov_from_buf(in_iov, in_num, 0, serial, size);

        virtio_blk_req_complete(req, VIRTIO_BLK_S_OK);

        virtio_blk_free_request(req);

        break;

    }

    default:

        virtio_blk_req_complete(req, VIRTIO_BLK_S_UNSUPP);

        virtio_blk_free_request(req);

    }

}

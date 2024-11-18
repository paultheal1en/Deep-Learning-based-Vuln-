static void coroutine_fn sd_write_done(SheepdogAIOCB *acb)

{

    BDRVSheepdogState *s = acb->common.bs->opaque;

    struct iovec iov;

    AIOReq *aio_req;

    uint32_t offset, data_len, mn, mx;



    mn = s->min_dirty_data_idx;

    mx = s->max_dirty_data_idx;

    if (mn <= mx) {

        /* we need to update the vdi object. */

        offset = sizeof(s->inode) - sizeof(s->inode.data_vdi_id) +

            mn * sizeof(s->inode.data_vdi_id[0]);

        data_len = (mx - mn + 1) * sizeof(s->inode.data_vdi_id[0]);



        s->min_dirty_data_idx = UINT32_MAX;

        s->max_dirty_data_idx = 0;



        iov.iov_base = &s->inode;

        iov.iov_len = sizeof(s->inode);

        aio_req = alloc_aio_req(s, acb, vid_to_vdi_oid(s->inode.vdi_id),

                                data_len, offset, 0, 0, offset);

        QLIST_INSERT_HEAD(&s->inflight_aio_head, aio_req, aio_siblings);

        add_aio_request(s, aio_req, &iov, 1, false, AIOCB_WRITE_UDATA);



        acb->aio_done_func = sd_finish_aiocb;

        acb->aiocb_type = AIOCB_WRITE_UDATA;

        return;

    }



    sd_finish_aiocb(acb);

}

static void coroutine_fn resend_aioreq(BDRVSheepdogState *s, AIOReq *aio_req)

{

    SheepdogAIOCB *acb = aio_req->aiocb;

    bool create = false;



    /* check whether this request becomes a CoW one */

    if (acb->aiocb_type == AIOCB_WRITE_UDATA && is_data_obj(aio_req->oid)) {

        int idx = data_oid_to_idx(aio_req->oid);



        if (is_data_obj_writable(&s->inode, idx)) {

            goto out;

        }



        if (check_simultaneous_create(s, aio_req)) {

            return;

        }



        if (s->inode.data_vdi_id[idx]) {

            aio_req->base_oid = vid_to_data_oid(s->inode.data_vdi_id[idx], idx);

            aio_req->flags |= SD_FLAG_CMD_COW;

        }

        create = true;

    }

out:

    if (is_data_obj(aio_req->oid)) {

        add_aio_request(s, aio_req, acb->qiov->iov, acb->qiov->niov, create,

                        acb->aiocb_type);

    } else {

        struct iovec iov;

        iov.iov_base = &s->inode;

        iov.iov_len = sizeof(s->inode);

        add_aio_request(s, aio_req, &iov, 1, false, AIOCB_WRITE_UDATA);

    }

}

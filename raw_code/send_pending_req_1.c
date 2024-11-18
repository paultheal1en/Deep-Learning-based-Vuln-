static void coroutine_fn send_pending_req(BDRVSheepdogState *s, uint64_t oid)

{

    AIOReq *aio_req;

    SheepdogAIOCB *acb;



    while ((aio_req = find_pending_req(s, oid)) != NULL) {

        acb = aio_req->aiocb;

        /* move aio_req from pending list to inflight one */

        QLIST_REMOVE(aio_req, aio_siblings);

        QLIST_INSERT_HEAD(&s->inflight_aio_head, aio_req, aio_siblings);

        add_aio_request(s, aio_req, acb->qiov->iov, acb->qiov->niov, false,

                        acb->aiocb_type);

    }

}

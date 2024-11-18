static inline AIOReq *alloc_aio_req(BDRVSheepdogState *s, SheepdogAIOCB *acb,

                                    uint64_t oid, unsigned int data_len,

                                    uint64_t offset, uint8_t flags,

                                    uint64_t base_oid, unsigned int iov_offset)

{

    AIOReq *aio_req;



    aio_req = g_malloc(sizeof(*aio_req));

    aio_req->aiocb = acb;

    aio_req->iov_offset = iov_offset;

    aio_req->oid = oid;

    aio_req->base_oid = base_oid;

    aio_req->offset = offset;

    aio_req->data_len = data_len;

    aio_req->flags = flags;

    aio_req->id = s->aioreq_seq_num++;



    acb->nr_pending++;

    return aio_req;

}

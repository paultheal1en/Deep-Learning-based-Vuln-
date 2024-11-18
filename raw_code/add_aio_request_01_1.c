static void coroutine_fn add_aio_request(BDRVSheepdogState *s, AIOReq *aio_req,

                           struct iovec *iov, int niov, bool create,

                           enum AIOCBState aiocb_type)

{

    int nr_copies = s->inode.nr_copies;

    SheepdogObjReq hdr;

    unsigned int wlen = 0;

    int ret;

    uint64_t oid = aio_req->oid;

    unsigned int datalen = aio_req->data_len;

    uint64_t offset = aio_req->offset;

    uint8_t flags = aio_req->flags;

    uint64_t old_oid = aio_req->base_oid;



    if (!nr_copies) {

        error_report("bug");

    }



    memset(&hdr, 0, sizeof(hdr));



    switch (aiocb_type) {

    case AIOCB_FLUSH_CACHE:

        hdr.opcode = SD_OP_FLUSH_VDI;

        break;

    case AIOCB_READ_UDATA:

        hdr.opcode = SD_OP_READ_OBJ;

        hdr.flags = flags;

        break;

    case AIOCB_WRITE_UDATA:

        if (create) {

            hdr.opcode = SD_OP_CREATE_AND_WRITE_OBJ;

        } else {

            hdr.opcode = SD_OP_WRITE_OBJ;

        }

        wlen = datalen;

        hdr.flags = SD_FLAG_CMD_WRITE | flags;

        break;

    case AIOCB_DISCARD_OBJ:

        hdr.opcode = SD_OP_DISCARD_OBJ;

        break;

    }



    if (s->cache_flags) {

        hdr.flags |= s->cache_flags;

    }



    hdr.oid = oid;

    hdr.cow_oid = old_oid;

    hdr.copies = s->inode.nr_copies;



    hdr.data_length = datalen;

    hdr.offset = offset;



    hdr.id = aio_req->id;



    qemu_co_mutex_lock(&s->lock);

    s->co_send = qemu_coroutine_self();

    aio_set_fd_handler(s->aio_context, s->fd,

                       co_read_response, co_write_request, s);

    socket_set_cork(s->fd, 1);



    /* send a header */

    ret = qemu_co_send(s->fd, &hdr, sizeof(hdr));

    if (ret != sizeof(hdr)) {

        error_report("failed to send a req, %s", strerror(errno));

        goto out;

    }



    if (wlen) {

        ret = qemu_co_sendv(s->fd, iov, niov, aio_req->iov_offset, wlen);

        if (ret != wlen) {

            error_report("failed to send a data, %s", strerror(errno));

        }

    }

out:

    socket_set_cork(s->fd, 0);

    aio_set_fd_handler(s->aio_context, s->fd, co_read_response, NULL, s);

    s->co_send = NULL;

    qemu_co_mutex_unlock(&s->lock);

}

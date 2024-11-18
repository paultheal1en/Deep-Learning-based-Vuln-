static int nbd_co_receive_offset_data_payload(NBDClientSession *s,

                                              uint64_t orig_offset,

                                              QEMUIOVector *qiov, Error **errp)

{

    QEMUIOVector sub_qiov;

    uint64_t offset;

    size_t data_size;

    int ret;

    NBDStructuredReplyChunk *chunk = &s->reply.structured;



    assert(nbd_reply_is_structured(&s->reply));



    if (chunk->length < sizeof(offset)) {

        error_setg(errp, "Protocol error: invalid payload for "

                         "NBD_REPLY_TYPE_OFFSET_DATA");

        return -EINVAL;

    }



    if (nbd_read(s->ioc, &offset, sizeof(offset), errp) < 0) {

        return -EIO;

    }

    be64_to_cpus(&offset);



    data_size = chunk->length - sizeof(offset);

    if (offset < orig_offset || data_size > qiov->size ||

        offset > orig_offset + qiov->size - data_size) {

        error_setg(errp, "Protocol error: server sent chunk exceeding requested"

                         " region");

        return -EINVAL;

    }



    qemu_iovec_init(&sub_qiov, qiov->niov);

    qemu_iovec_concat(&sub_qiov, qiov, offset - orig_offset, data_size);

    ret = qio_channel_readv_all(s->ioc, sub_qiov.iov, sub_qiov.niov, errp);

    qemu_iovec_destroy(&sub_qiov);



    return ret < 0 ? -EIO : 0;

}

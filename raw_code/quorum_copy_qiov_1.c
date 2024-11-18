static void quorum_copy_qiov(QEMUIOVector *dest, QEMUIOVector *source)

{

    int i;

    assert(dest->niov == source->niov);

    assert(dest->size == source->size);

    for (i = 0; i < source->niov; i++) {

        assert(dest->iov[i].iov_len == source->iov[i].iov_len);

        memcpy(dest->iov[i].iov_base,

               source->iov[i].iov_base,

               source->iov[i].iov_len);

    }

}

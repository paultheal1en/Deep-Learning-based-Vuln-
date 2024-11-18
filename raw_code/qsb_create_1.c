QEMUSizedBuffer *qsb_create(const uint8_t *buffer, size_t len)

{

    QEMUSizedBuffer *qsb;

    size_t alloc_len, num_chunks, i, to_copy;

    size_t chunk_size = (len > QSB_MAX_CHUNK_SIZE)

                        ? QSB_MAX_CHUNK_SIZE

                        : QSB_CHUNK_SIZE;



    num_chunks = DIV_ROUND_UP(len ? len : QSB_CHUNK_SIZE, chunk_size);

    alloc_len = num_chunks * chunk_size;



    qsb = g_try_new0(QEMUSizedBuffer, 1);

    if (!qsb) {

        return NULL;

    }



    qsb->iov = g_try_new0(struct iovec, num_chunks);

    if (!qsb->iov) {

        g_free(qsb);

        return NULL;

    }



    qsb->n_iov = num_chunks;



    for (i = 0; i < num_chunks; i++) {

        qsb->iov[i].iov_base = g_try_malloc0(chunk_size);

        if (!qsb->iov[i].iov_base) {

            /* qsb_free is safe since g_free can cope with NULL */

            qsb_free(qsb);

            return NULL;

        }



        qsb->iov[i].iov_len = chunk_size;

        if (buffer) {

            to_copy = (len - qsb->used) > chunk_size

                      ? chunk_size : (len - qsb->used);

            memcpy(qsb->iov[i].iov_base, &buffer[qsb->used], to_copy);

            qsb->used += to_copy;

        }

    }



    qsb->size = alloc_len;



    return qsb;

}

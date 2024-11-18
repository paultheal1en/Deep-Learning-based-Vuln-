static ssize_t qsb_grow(QEMUSizedBuffer *qsb, size_t new_size)

{

    size_t needed_chunks, i;



    if (qsb->size < new_size) {

        struct iovec *new_iov;

        size_t size_diff = new_size - qsb->size;

        size_t chunk_size = (size_diff > QSB_MAX_CHUNK_SIZE)

                             ? QSB_MAX_CHUNK_SIZE : QSB_CHUNK_SIZE;



        needed_chunks = DIV_ROUND_UP(size_diff, chunk_size);



        new_iov = g_try_new(struct iovec, qsb->n_iov + needed_chunks);

        if (new_iov == NULL) {

            return -ENOMEM;

        }



        /* Allocate new chunks as needed into new_iov */

        for (i = qsb->n_iov; i < qsb->n_iov + needed_chunks; i++) {

            new_iov[i].iov_base = g_try_malloc0(chunk_size);

            new_iov[i].iov_len = chunk_size;

            if (!new_iov[i].iov_base) {

                size_t j;



                /* Free previously allocated new chunks */

                for (j = qsb->n_iov; j < i; j++) {

                    g_free(new_iov[j].iov_base);

                }

                g_free(new_iov);



                return -ENOMEM;

            }

        }



        /*

         * Now we can't get any allocation errors, copy over to new iov

         * and switch.

         */

        for (i = 0; i < qsb->n_iov; i++) {

            new_iov[i] = qsb->iov[i];

        }



        qsb->n_iov += needed_chunks;

        g_free(qsb->iov);

        qsb->iov = new_iov;

        qsb->size += (needed_chunks * chunk_size);

    }



    return qsb->size;

}

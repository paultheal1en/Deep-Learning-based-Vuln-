QEMUSizedBuffer *qsb_clone(const QEMUSizedBuffer *qsb)

{

    QEMUSizedBuffer *out = qsb_create(NULL, qsb_get_length(qsb));

    size_t i;

    ssize_t res;

    off_t pos = 0;



    if (!out) {

        return NULL;

    }



    for (i = 0; i < qsb->n_iov; i++) {

        res =  qsb_write_at(out, qsb->iov[i].iov_base,

                            pos, qsb->iov[i].iov_len);

        if (res < 0) {

            qsb_free(out);

            return NULL;

        }

        pos += res;

    }



    return out;

}

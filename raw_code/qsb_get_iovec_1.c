static ssize_t qsb_get_iovec(const QEMUSizedBuffer *qsb,

                             off_t pos, off_t *d_off)

{

    ssize_t i;

    off_t curr = 0;



    if (pos > qsb->used) {

        return -1;

    }



    for (i = 0; i < qsb->n_iov; i++) {

        if (curr + qsb->iov[i].iov_len > pos) {

            *d_off = pos - curr;

            return i;

        }

        curr += qsb->iov[i].iov_len;

    }

    return -1;

}

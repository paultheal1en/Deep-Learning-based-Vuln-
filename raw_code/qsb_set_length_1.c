size_t qsb_set_length(QEMUSizedBuffer *qsb, size_t new_len)

{

    if (new_len <= qsb->size) {

        qsb->used = new_len;

    } else {

        qsb->used = qsb->size;

    }

    return qsb->used;

}

static uint8_t *csrhci_out_packet(struct csrhci_s *s, int len)

{

    int off = s->out_start + s->out_len;



    /* TODO: do the padding here, i.e. align len */

    s->out_len += len;



    if (off < FIFO_LEN) {

        if (off + len > FIFO_LEN && (s->out_size = off + len) > FIFO_LEN * 2) {

            fprintf(stderr, "%s: can't alloc %i bytes\n", __func__, len);

            exit(-1);

        }

        return s->outfifo + off;

    }



    if (s->out_len > s->out_size) {

        fprintf(stderr, "%s: can't alloc %i bytes\n", __func__, len);

        exit(-1);

    }



    return s->outfifo + off - s->out_size;

}

static void receive_from_chr_layer(SCLPConsole *scon, const uint8_t *buf,

                                   int size)

{

    /* read data must fit into current buffer */

    assert(size <= SIZE_BUFFER_VT220 - scon->iov_data_len);



    /* put byte-stream from character layer into buffer */

    memcpy(&scon->iov[scon->iov_bs], buf, size);

    scon->iov_data_len += size;

    scon->iov_sclp_rest += size;

    scon->iov_bs += size;

    scon->event.event_pending = true;

}

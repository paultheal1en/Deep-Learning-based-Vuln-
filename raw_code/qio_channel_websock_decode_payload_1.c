static ssize_t qio_channel_websock_decode_payload(QIOChannelWebsock *ioc,

                                                  Error **errp)

{

    size_t i;

    size_t payload_len;

    uint32_t *payload32;



    if (!ioc->payload_remain) {

        error_setg(errp,

                   "Decoding payload but no bytes of payload remain");

        return -1;

    }



    /* If we aren't at the end of the payload, then drop

     * off the last bytes, so we're always multiple of 4

     * for purpose of unmasking, except at end of payload

     */

    if (ioc->encinput.offset < ioc->payload_remain) {

        payload_len = ioc->encinput.offset - (ioc->encinput.offset % 4);

    } else {

        payload_len = ioc->payload_remain;

    }

    if (payload_len == 0) {

        return QIO_CHANNEL_ERR_BLOCK;

    }



    ioc->payload_remain -= payload_len;



    /* unmask frame */

    /* process 1 frame (32 bit op) */

    payload32 = (uint32_t *)ioc->encinput.buffer;

    for (i = 0; i < payload_len / 4; i++) {

        payload32[i] ^= ioc->mask.u;

    }

    /* process the remaining bytes (if any) */

    for (i *= 4; i < payload_len; i++) {

        ioc->encinput.buffer[i] ^= ioc->mask.c[i % 4];

    }



    buffer_reserve(&ioc->rawinput, payload_len);

    buffer_append(&ioc->rawinput, ioc->encinput.buffer, payload_len);

    buffer_advance(&ioc->encinput, payload_len);

    return payload_len;

}

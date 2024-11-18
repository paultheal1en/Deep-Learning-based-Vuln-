static int qio_channel_websock_handshake_read(QIOChannelWebsock *ioc,

                                              Error **errp)

{

    char *handshake_end;

    ssize_t ret;

    /* Typical HTTP headers from novnc are 512 bytes, so limiting

     * total header size to 4096 is easily enough. */

    size_t want = 4096 - ioc->encinput.offset;

    buffer_reserve(&ioc->encinput, want);

    ret = qio_channel_read(ioc->master,

                           (char *)buffer_end(&ioc->encinput), want, errp);

    if (ret < 0) {

        return -1;

    }

    ioc->encinput.offset += ret;



    handshake_end = g_strstr_len((char *)ioc->encinput.buffer,

                                 ioc->encinput.offset,

                                 QIO_CHANNEL_WEBSOCK_HANDSHAKE_END);

    if (!handshake_end) {

        if (ioc->encinput.offset >= 4096) {

            error_setg(errp,

                       "End of headers not found in first 4096 bytes");

            return -1;

        } else {

            return 0;

        }

    }

    *handshake_end = '\0';



    if (qio_channel_websock_handshake_process(ioc,

                                              (char *)ioc->encinput.buffer,

                                              errp) < 0) {

        return -1;

    }



    buffer_advance(&ioc->encinput,

                   handshake_end - (char *)ioc->encinput.buffer +

                   strlen(QIO_CHANNEL_WEBSOCK_HANDSHAKE_END));

    return 1;

}

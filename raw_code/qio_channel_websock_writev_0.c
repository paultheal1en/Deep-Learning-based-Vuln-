static ssize_t qio_channel_websock_writev(QIOChannel *ioc,

                                          const struct iovec *iov,

                                          size_t niov,

                                          int *fds,

                                          size_t nfds,

                                          Error **errp)

{

    QIOChannelWebsock *wioc = QIO_CHANNEL_WEBSOCK(ioc);

    size_t i;

    ssize_t done = 0;

    ssize_t ret;



    if (wioc->io_err) {

        *errp = error_copy(wioc->io_err);

        return -1;

    }



    if (wioc->io_eof) {

        error_setg(errp, "%s", "Broken pipe");

        return -1;

    }



    for (i = 0; i < niov; i++) {

        size_t want = iov[i].iov_len;

        if ((want + wioc->rawoutput.offset) > QIO_CHANNEL_WEBSOCK_MAX_BUFFER) {

            want = (QIO_CHANNEL_WEBSOCK_MAX_BUFFER - wioc->rawoutput.offset);

        }

        if (want == 0) {

            goto done;

        }



        buffer_reserve(&wioc->rawoutput, want);

        buffer_append(&wioc->rawoutput, iov[i].iov_base, want);

        done += want;

        if (want < iov[i].iov_len) {

            break;

        }

    }



 done:

    ret = qio_channel_websock_write_wire(wioc, errp);

    if (ret < 0 &&

        ret != QIO_CHANNEL_ERR_BLOCK) {

        qio_channel_websock_unset_watch(wioc);

        return -1;

    }



    qio_channel_websock_set_watch(wioc);



    if (done == 0) {

        return QIO_CHANNEL_ERR_BLOCK;

    }



    return done;

}

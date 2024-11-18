static ssize_t qio_channel_command_writev(QIOChannel *ioc,

                                          const struct iovec *iov,

                                          size_t niov,

                                          int *fds,

                                          size_t nfds,

                                          Error **errp)

{

    QIOChannelCommand *cioc = QIO_CHANNEL_COMMAND(ioc);

    ssize_t ret;



 retry:

    ret = writev(cioc->writefd, iov, niov);

    if (ret <= 0) {

        if (errno == EAGAIN ||

            errno == EWOULDBLOCK) {

            return QIO_CHANNEL_ERR_BLOCK;

        }

        if (errno == EINTR) {

            goto retry;

        }

        error_setg_errno(errp, errno, "%s",

                         "Unable to write to command");

        return -1;

    }

    return ret;

}

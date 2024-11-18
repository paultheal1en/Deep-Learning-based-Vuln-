static int io_channel_send_full(QIOChannel *ioc,

                                const void *buf, size_t len,

                                int *fds, size_t nfds)

{

    size_t offset = 0;



    while (offset < len) {

        ssize_t ret = 0;

        struct iovec iov = { .iov_base = (char *)buf + offset,

                             .iov_len = len - offset };



        ret = qio_channel_writev_full(

            ioc, &iov, 1,

            fds, nfds, NULL);

        if (ret == QIO_CHANNEL_ERR_BLOCK) {

            errno = EAGAIN;

            return -1;

        } else if (ret < 0) {

            if (offset) {

                return offset;

            }



            errno = EINVAL;

            return -1;

        }



        offset += ret;

    }



    return offset;

}

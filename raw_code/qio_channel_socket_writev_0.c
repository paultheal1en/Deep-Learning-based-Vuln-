static ssize_t qio_channel_socket_writev(QIOChannel *ioc,

                                         const struct iovec *iov,

                                         size_t niov,

                                         int *fds,

                                         size_t nfds,

                                         Error **errp)

{

    QIOChannelSocket *sioc = QIO_CHANNEL_SOCKET(ioc);

    ssize_t ret;

    struct msghdr msg = { NULL, };

    char control[CMSG_SPACE(sizeof(int) * SOCKET_MAX_FDS)] = { 0 };

    size_t fdsize = sizeof(int) * nfds;

    struct cmsghdr *cmsg;



    msg.msg_iov = (struct iovec *)iov;

    msg.msg_iovlen = niov;



    if (nfds) {

        if (nfds > SOCKET_MAX_FDS) {

            error_setg_errno(errp, EINVAL,

                             "Only %d FDs can be sent, got %zu",

                             SOCKET_MAX_FDS, nfds);

            return -1;

        }



        msg.msg_control = control;

        msg.msg_controllen = CMSG_SPACE(sizeof(int) * nfds);



        cmsg = CMSG_FIRSTHDR(&msg);

        cmsg->cmsg_len = CMSG_LEN(fdsize);

        cmsg->cmsg_level = SOL_SOCKET;

        cmsg->cmsg_type = SCM_RIGHTS;

        memcpy(CMSG_DATA(cmsg), fds, fdsize);

    }



 retry:

    ret = sendmsg(sioc->fd, &msg, 0);

    if (ret <= 0) {

        if (socket_error() == EAGAIN ||

            socket_error() == EWOULDBLOCK) {

            return QIO_CHANNEL_ERR_BLOCK;

        }

        if (socket_error() == EINTR) {

            goto retry;

        }

        error_setg_errno(errp, socket_error(),

                         "Unable to write to socket");

        return -1;

    }

    return ret;

}

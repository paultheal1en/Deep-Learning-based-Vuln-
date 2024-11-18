static int tcp_write(URLContext *h, const uint8_t *buf, int size)

{

    TCPContext *s = h->priv_data;

    int ret, size1, fd_max, len;

    fd_set wfds;

    struct timeval tv;



    size1 = size;

    while (size > 0) {

        if (url_interrupt_cb())

            return AVERROR(EINTR);

        fd_max = s->fd;

        FD_ZERO(&wfds);

        FD_SET(s->fd, &wfds);

        tv.tv_sec = 0;

        tv.tv_usec = 100 * 1000;

        ret = select(fd_max + 1, NULL, &wfds, NULL, &tv);

        if (ret > 0 && FD_ISSET(s->fd, &wfds)) {

            len = send(s->fd, buf, size, 0);

            if (len < 0) {

                if (ff_neterrno() != FF_NETERROR(EINTR) &&

                    ff_neterrno() != FF_NETERROR(EAGAIN))

                    return ff_neterrno();

                continue;

            }

            size -= len;

            buf += len;

        } else if (ret < 0) {

            if (ff_neterrno() == FF_NETERROR(EINTR))

                continue;

            return -1;

        }

    }

    return size1 - size;

}

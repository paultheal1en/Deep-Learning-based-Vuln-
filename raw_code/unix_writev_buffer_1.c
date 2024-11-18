static ssize_t unix_writev_buffer(void *opaque, struct iovec *iov, int iovcnt,

                                  int64_t pos)

{

    QEMUFileSocket *s = opaque;

    ssize_t len, offset;

    ssize_t size = iov_size(iov, iovcnt);

    ssize_t total = 0;



    assert(iovcnt > 0);

    offset = 0;

    while (size > 0) {

        /* Find the next start position; skip all full-sized vector elements  */

        while (offset >= iov[0].iov_len) {

            offset -= iov[0].iov_len;

            iov++, iovcnt--;

        }



        /* skip `offset' bytes from the (now) first element, undo it on exit */

        assert(iovcnt > 0);

        iov[0].iov_base += offset;

        iov[0].iov_len -= offset;



        do {

            len = writev(s->fd, iov, iovcnt);

        } while (len == -1 && errno == EINTR);

        if (len == -1) {

            return -errno;

        }



        /* Undo the changes above */

        iov[0].iov_base -= offset;

        iov[0].iov_len += offset;



        /* Prepare for the next iteration */

        offset += len;

        total += len;

        size -= len;

    }



    return total;

}

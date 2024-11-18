ssize_t nbd_wr_sync(int fd, void *buffer, size_t size, bool do_read)

{

    size_t offset = 0;

    int err;



    if (qemu_in_coroutine()) {

        if (do_read) {

            return qemu_co_recv(fd, buffer, size);

        } else {

            return qemu_co_send(fd, buffer, size);

        }

    }



    while (offset < size) {

        ssize_t len;



        if (do_read) {

            len = qemu_recv(fd, buffer + offset, size - offset, 0);

        } else {

            len = send(fd, buffer + offset, size - offset, 0);

        }



        if (len < 0) {

            err = socket_error();



            /* recoverable error */

            if (err == EINTR || err == EAGAIN) {

                continue;

            }



            /* unrecoverable error */

            return -err;

        }



        /* eof */

        if (len == 0) {

            break;

        }



        offset += len;

    }



    return offset;

}

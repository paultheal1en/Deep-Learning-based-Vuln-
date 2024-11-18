static int get_char(GDBState *s)

{

    uint8_t ch;

    int ret;



    for(;;) {

        ret = qemu_recv(s->fd, &ch, 1, 0);

        if (ret < 0) {

            if (errno == ECONNRESET)

                s->fd = -1;

            if (errno != EINTR && errno != EAGAIN)

                return -1;

        } else if (ret == 0) {

            close(s->fd);

            s->fd = -1;

            return -1;

        } else {

            break;

        }

    }

    return ch;

}

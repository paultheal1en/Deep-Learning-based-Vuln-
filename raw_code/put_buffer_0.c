static void put_buffer(GDBState *s, const uint8_t *buf, int len)

{

#ifdef CONFIG_USER_ONLY

    int ret;



    while (len > 0) {

        ret = send(s->fd, buf, len, 0);

        if (ret < 0) {

            if (errno != EINTR && errno != EAGAIN)

                return;

        } else {

            buf += ret;

            len -= ret;

        }

    }

#else

    qemu_chr_fe_write(s->chr, buf, len);

#endif

}

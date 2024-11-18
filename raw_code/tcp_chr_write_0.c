static int tcp_chr_write(CharDriverState *chr, const uint8_t *buf, int len)

{

    TCPCharDriver *s = chr->opaque;

    if (s->connected) {

        return send_all(s->fd, buf, len);

    } else {

        /* (Re-)connect for unconnected writing */

        tcp_chr_connect(chr);

        return 0;

    }

}

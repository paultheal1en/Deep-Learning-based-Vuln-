static void net_socket_send_dgram(void *opaque)

{

    NetSocketState *s = opaque;

    int size;



    size = qemu_recv(s->fd, s->buf, sizeof(s->buf), 0);

    if (size < 0)

        return;

    if (size == 0) {

        /* end of connection */

        net_socket_read_poll(s, false);

        net_socket_write_poll(s, false);

        return;

    }

    qemu_send_packet(&s->nc, s->buf, size);

}

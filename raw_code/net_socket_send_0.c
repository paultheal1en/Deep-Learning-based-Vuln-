static void net_socket_send(void *opaque)

{

    NetSocketState *s = opaque;

    int size, err;

    unsigned l;

    uint8_t buf1[NET_BUFSIZE];

    const uint8_t *buf;



    size = qemu_recv(s->fd, buf1, sizeof(buf1), 0);

    if (size < 0) {

        err = socket_error();

        if (err != EWOULDBLOCK)

            goto eoc;

    } else if (size == 0) {

        /* end of connection */

    eoc:

        net_socket_read_poll(s, false);

        net_socket_write_poll(s, false);

        if (s->listen_fd != -1) {

            qemu_set_fd_handler(s->listen_fd, net_socket_accept, NULL, s);

        }

        closesocket(s->fd);



        s->fd = -1;

        s->state = 0;

        s->index = 0;

        s->packet_len = 0;

        s->nc.link_down = true;

        memset(s->buf, 0, sizeof(s->buf));

        memset(s->nc.info_str, 0, sizeof(s->nc.info_str));



        return;

    }

    buf = buf1;

    while (size > 0) {

        /* reassemble a packet from the network */

        switch(s->state) {

        case 0:

            l = 4 - s->index;

            if (l > size)

                l = size;

            memcpy(s->buf + s->index, buf, l);

            buf += l;

            size -= l;

            s->index += l;

            if (s->index == 4) {

                /* got length */

                s->packet_len = ntohl(*(uint32_t *)s->buf);

                s->index = 0;

                s->state = 1;

            }

            break;

        case 1:

            l = s->packet_len - s->index;

            if (l > size)

                l = size;

            if (s->index + l <= sizeof(s->buf)) {

                memcpy(s->buf + s->index, buf, l);

            } else {

                fprintf(stderr, "serious error: oversized packet received,"

                    "connection terminated.\n");

                s->state = 0;

                goto eoc;

            }



            s->index += l;

            buf += l;

            size -= l;

            if (s->index >= s->packet_len) {

                qemu_send_packet(&s->nc, s->buf, s->packet_len);

                s->index = 0;

                s->state = 0;

            }

            break;

        }

    }

}

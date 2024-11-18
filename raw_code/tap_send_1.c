static void tap_send(void *opaque)

{

    TAPState *s = opaque;

    int size;



    do {

        uint8_t *buf = s->buf;



        size = tap_read_packet(s->fd, s->buf, sizeof(s->buf));

        if (size <= 0) {

            break;

        }



        if (s->host_vnet_hdr_len && !s->using_vnet_hdr) {

            buf  += s->host_vnet_hdr_len;

            size -= s->host_vnet_hdr_len;

        }



        size = qemu_send_packet_async(&s->nc, buf, size, tap_send_completed);

        if (size == 0) {

            tap_read_poll(s, false);

        }

    } while (size > 0 && qemu_can_send_packet(&s->nc));

}

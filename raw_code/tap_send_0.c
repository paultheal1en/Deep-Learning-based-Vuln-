static void tap_send(void *opaque)

{

    TAPState *s = opaque;

    int size;

    int packets = 0;



    while (qemu_can_send_packet(&s->nc)) {

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

            break;

        } else if (size < 0) {

            break;

        }



        /*

         * When the host keeps receiving more packets while tap_send() is

         * running we can hog the QEMU global mutex.  Limit the number of

         * packets that are processed per tap_send() callback to prevent

         * stalling the guest.

         */

        packets++;

        if (packets >= 50) {

            break;

        }

    }

}

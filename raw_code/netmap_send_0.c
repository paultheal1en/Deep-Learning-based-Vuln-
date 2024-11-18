static void netmap_send(void *opaque)

{

    NetmapState *s = opaque;

    struct netmap_ring *ring = s->me.rx;



    /* Keep sending while there are available packets into the netmap

       RX ring and the forwarding path towards the peer is open. */

    while (!nm_ring_empty(ring) && qemu_can_send_packet(&s->nc)) {

        uint32_t i;

        uint32_t idx;

        bool morefrag;

        int iovcnt = 0;

        int iovsize;



        do {

            i = ring->cur;

            idx = ring->slot[i].buf_idx;

            morefrag = (ring->slot[i].flags & NS_MOREFRAG);

            s->iov[iovcnt].iov_base = (u_char *)NETMAP_BUF(ring, idx);

            s->iov[iovcnt].iov_len = ring->slot[i].len;

            iovcnt++;



            ring->cur = ring->head = nm_ring_next(ring, i);

        } while (!nm_ring_empty(ring) && morefrag);



        if (unlikely(nm_ring_empty(ring) && morefrag)) {

            RD(5, "[netmap_send] ran out of slots, with a pending"

                   "incomplete packet\n");

        }



        iovsize = qemu_sendv_packet_async(&s->nc, s->iov, iovcnt,

                                            netmap_send_completed);



        if (iovsize == 0) {

            /* The peer does not receive anymore. Packet is queued, stop

             * reading from the backend until netmap_send_completed()

             */

            netmap_read_poll(s, false);

            break;

        }

    }

}

static ssize_t colo_rewriter_receive_iov(NetFilterState *nf,

                                         NetClientState *sender,

                                         unsigned flags,

                                         const struct iovec *iov,

                                         int iovcnt,

                                         NetPacketSent *sent_cb)

{

    RewriterState *s = FILTER_COLO_REWRITER(nf);

    Connection *conn;

    ConnectionKey key;

    Packet *pkt;

    ssize_t size = iov_size(iov, iovcnt);

    char *buf = g_malloc0(size);



    iov_to_buf(iov, iovcnt, 0, buf, size);

    pkt = packet_new(buf, size);



    /*

     * if we get tcp packet

     * we will rewrite it to make secondary guest's

     * connection established successfully

     */

    if (pkt && is_tcp_packet(pkt)) {



        fill_connection_key(pkt, &key);



        if (sender == nf->netdev) {

            /*

             * We need make tcp TX and RX packet

             * into one connection.

             */

            reverse_connection_key(&key);

        }

        conn = connection_get(s->connection_track_table,

                              &key,

                              NULL);



        if (sender == nf->netdev) {

            /* NET_FILTER_DIRECTION_TX */

            /* handle_primary_tcp_pkt */

        } else {

            /* NET_FILTER_DIRECTION_RX */

            /* handle_secondary_tcp_pkt */

        }

    }



    packet_destroy(pkt, NULL);

    pkt = NULL;

    return 0;

}

static void net_l2tpv3_cleanup(NetClientState *nc)

{

    NetL2TPV3State *s = DO_UPCAST(NetL2TPV3State, nc, nc);

    qemu_purge_queued_packets(nc);

    l2tpv3_read_poll(s, false);

    l2tpv3_write_poll(s, false);

    if (s->fd > 0) {

        close(s->fd);

    }

    destroy_vector(s->msgvec, MAX_L2TPV3_MSGCNT, IOVSIZE);

    g_free(s->vec);

    g_free(s->header_buf);

    g_free(s->dgram_dst);

}

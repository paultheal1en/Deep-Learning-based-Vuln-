static void compare_pri_rs_finalize(SocketReadState *pri_rs)

{

    CompareState *s = container_of(pri_rs, CompareState, pri_rs);



    if (packet_enqueue(s, PRIMARY_IN)) {

        trace_colo_compare_main("primary: unsupported packet in");

        compare_chr_send(s,

                         pri_rs->buf,

                         pri_rs->packet_len,

                         pri_rs->vnet_hdr_len);

    } else {

        /* compare connection */

        g_queue_foreach(&s->conn_list, colo_compare_connection, s);

    }

}

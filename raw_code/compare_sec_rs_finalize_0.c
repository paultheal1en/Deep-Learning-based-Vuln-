static void compare_sec_rs_finalize(SocketReadState *sec_rs)

{

    CompareState *s = container_of(sec_rs, CompareState, sec_rs);



    if (packet_enqueue(s, SECONDARY_IN)) {

        trace_colo_compare_main("secondary: unsupported packet in");

    } else {

        /* compare connection */

        g_queue_foreach(&s->conn_list, colo_compare_connection, s);

    }

}

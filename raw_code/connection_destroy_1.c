void connection_destroy(void *opaque)

{

    Connection *conn = opaque;



    g_queue_foreach(&conn->primary_list, packet_destroy, NULL);

    g_queue_free(&conn->primary_list);

    g_queue_foreach(&conn->secondary_list, packet_destroy, NULL);

    g_queue_free(&conn->secondary_list);

    g_slice_free(Connection, conn);

}

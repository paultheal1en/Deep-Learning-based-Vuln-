static void colo_old_packet_check_one_conn(void *opaque,

                                           void *user_data)

{

    Connection *conn = opaque;

    GList *result = NULL;

    int64_t check_time = REGULAR_PACKET_CHECK_MS;



    result = g_queue_find_custom(&conn->primary_list,

                                 &check_time,

                                 (GCompareFunc)colo_old_packet_check_one);



    if (result) {

        /* do checkpoint will flush old packet */

        /* TODO: colo_notify_checkpoint();*/

    }

}

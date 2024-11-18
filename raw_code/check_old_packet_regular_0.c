static gboolean check_old_packet_regular(void *opaque)

{

    CompareState *s = opaque;



    /* if have old packet we will notify checkpoint */

    colo_old_packet_check(s);



    return TRUE;

}

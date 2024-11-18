static int l2tpv3_can_send(void *opaque)

{

    NetL2TPV3State *s = opaque;



    return qemu_can_send_packet(&s->nc);

}

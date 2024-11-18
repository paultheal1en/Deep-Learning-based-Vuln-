static int tap_can_send(void *opaque)

{

    TAPState *s = opaque;



    return qemu_can_send_packet(&s->nc);

}

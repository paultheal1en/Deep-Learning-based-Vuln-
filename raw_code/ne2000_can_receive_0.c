static int ne2000_can_receive(void *opaque)

{

    NE2000State *s = opaque;



    if (s->cmd & E8390_STOP)

        return 1;

    return !ne2000_buffer_full(s);

}

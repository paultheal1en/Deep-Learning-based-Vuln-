int ne2000_can_receive(NetClientState *nc)

{

    NE2000State *s = qemu_get_nic_opaque(nc);



    if (s->cmd & E8390_STOP)

        return 1;

    return !ne2000_buffer_full(s);

}

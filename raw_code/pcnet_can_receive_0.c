int pcnet_can_receive(NetClientState *nc)

{

    PCNetState *s = qemu_get_nic_opaque(nc);

    if (CSR_STOP(s) || CSR_SPND(s))

        return 0;



    return sizeof(s->buffer)-16;

}

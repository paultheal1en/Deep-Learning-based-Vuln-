e1000_can_receive(NetClientState *nc)

{

    E1000State *s = qemu_get_nic_opaque(nc);



    return (s->mac_reg[RCTL] & E1000_RCTL_EN) && e1000_has_rxbufs(s, 1);

}

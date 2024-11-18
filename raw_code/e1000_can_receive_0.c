e1000_can_receive(void *opaque)

{

    E1000State *s = opaque;



    return (s->mac_reg[RCTL] & E1000_RCTL_EN);

}

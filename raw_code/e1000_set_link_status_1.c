e1000_set_link_status(VLANClientState *nc)

{

    E1000State *s = DO_UPCAST(NICState, nc, nc)->opaque;

    uint32_t old_status = s->mac_reg[STATUS];



    if (nc->link_down)

        s->mac_reg[STATUS] &= ~E1000_STATUS_LU;

    else

        s->mac_reg[STATUS] |= E1000_STATUS_LU;



    if (s->mac_reg[STATUS] != old_status)

        set_ics(s, 0, E1000_ICR_LSC);

}

static void nic_reset(void *opaque)

{

    EEPRO100State *s = opaque;

    TRACE(OTHER, logout("%p\n", s));

    /* TODO: Clearing of multicast table for selective reset, too? */

    memset(&s->mult[0], 0, sizeof(s->mult));

    nic_selective_reset(s);

}

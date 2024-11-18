static void rtl8139_cleanup(NetClientState *nc)

{

    RTL8139State *s = qemu_get_nic_opaque(nc);



    s->nic = NULL;

}

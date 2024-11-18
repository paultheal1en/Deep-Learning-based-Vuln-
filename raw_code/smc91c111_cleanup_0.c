static void smc91c111_cleanup(NetClientState *nc)

{

    smc91c111_state *s = qemu_get_nic_opaque(nc);



    s->nic = NULL;

}

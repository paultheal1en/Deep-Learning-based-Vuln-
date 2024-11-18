static void gem_cleanup(NetClientState *nc)

{

    GemState *s = qemu_get_nic_opaque(nc);



    DB_PRINT("\n");

    s->nic = NULL;

}

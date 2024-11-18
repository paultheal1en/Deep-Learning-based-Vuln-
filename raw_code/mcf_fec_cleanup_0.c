static void mcf_fec_cleanup(NetClientState *nc)

{

    mcf_fec_state *s = qemu_get_nic_opaque(nc);



    g_free(s);

}

static void lance_cleanup(NetClientState *nc)

{

    PCNetState *d = qemu_get_nic_opaque(nc);



    pcnet_common_cleanup(d);

}

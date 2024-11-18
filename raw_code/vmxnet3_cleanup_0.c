static void vmxnet3_cleanup(NetClientState *nc)

{

    VMXNET3State *s = qemu_get_nic_opaque(nc);

    s->nic = NULL;

}

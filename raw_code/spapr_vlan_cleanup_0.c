static void spapr_vlan_cleanup(NetClientState *nc)

{

    VIOsPAPRVLANDevice *dev = qemu_get_nic_opaque(nc);



    dev->nic = NULL;

}

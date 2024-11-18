static void bt_vhci_add(int vlan_id)

{

    struct bt_scatternet_s *vlan = qemu_find_bt_vlan(vlan_id);



    if (!vlan->slave)

        fprintf(stderr, "qemu: warning: adding a VHCI to "

                        "an empty scatternet %i\n", vlan_id);



    bt_vhci_init(bt_new_hci(vlan));

}

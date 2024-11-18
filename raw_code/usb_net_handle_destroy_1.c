static void usb_net_handle_destroy(USBDevice *dev)

{

    USBNetState *s = (USBNetState *) dev;



    /* TODO: remove the nd_table[] entry */

    qemu_del_vlan_client(s->vc);

    rndis_clear_responsequeue(s);

    qemu_free(s);

}

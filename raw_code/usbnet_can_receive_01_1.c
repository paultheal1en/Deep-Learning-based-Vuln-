static int usbnet_can_receive(NetClientState *nc)
{
    USBNetState *s = qemu_get_nic_opaque(nc);
    if (is_rndis(s) && s->rndis_state != RNDIS_DATA_INITIALIZED) {
        return 1;
    return !s->in_len;
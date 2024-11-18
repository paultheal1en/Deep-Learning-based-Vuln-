static int usbnet_can_receive(void *opaque)

{

    USBNetState *s = opaque;



    if (s->rndis && !s->rndis_state == RNDIS_DATA_INITIALIZED)

        return 1;



    return !s->in_len;

}

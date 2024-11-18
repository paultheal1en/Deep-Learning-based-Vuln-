static int bt_hid_in(struct bt_hid_device_s *s)

{

    USBPacket p;



    p.pid = USB_TOKEN_IN;

    p.devep = 1;

    p.data = s->datain.buffer;

    p.len = sizeof(s->datain.buffer);

    s->datain.len = s->usbdev->info->handle_data(s->usbdev, &p);



    return s->datain.len;

}

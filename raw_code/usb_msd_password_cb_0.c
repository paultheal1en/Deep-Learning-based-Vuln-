static void usb_msd_password_cb(void *opaque, int err)

{

    MSDState *s = opaque;



    if (!err)

        err = usb_device_attach(&s->dev);



    if (err)

        qdev_unplug(&s->dev.qdev, NULL);

}

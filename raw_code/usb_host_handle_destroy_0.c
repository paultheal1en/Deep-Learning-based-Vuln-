static void usb_host_handle_destroy(USBDevice *udev)

{

    USBHostDevice *s = USB_HOST_DEVICE(udev);



    qemu_remove_exit_notifier(&s->exit);

    QTAILQ_REMOVE(&hostdevs, s, next);

    usb_host_close(s);

}

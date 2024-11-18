static void usbredir_handle_destroy(USBDevice *udev)

{

    USBRedirDevice *dev = DO_UPCAST(USBRedirDevice, dev, udev);



    qemu_chr_delete(dev->cs);


    /* Note must be done after qemu_chr_close, as that causes a close event */

    qemu_bh_delete(dev->chardev_close_bh);



    qemu_del_timer(dev->attach_timer);

    qemu_free_timer(dev->attach_timer);



    usbredir_cleanup_device_queues(dev);



    if (dev->parser) {

        usbredirparser_destroy(dev->parser);

    }

    if (dev->watch) {

        g_source_remove(dev->watch);

    }



    free(dev->filter_rules);

}
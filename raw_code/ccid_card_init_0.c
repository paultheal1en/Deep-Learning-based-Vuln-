static int ccid_card_init(DeviceState *qdev)

{

    CCIDCardState *card = CCID_CARD(qdev);

    USBDevice *dev = USB_DEVICE(qdev->parent_bus->parent);

    USBCCIDState *s = USB_CCID_DEV(dev);

    int ret = 0;



    if (card->slot != 0) {

        error_report("Warning: usb-ccid supports one slot, can't add %d",

                card->slot);

        return -1;

    }

    if (s->card != NULL) {

        error_report("Warning: usb-ccid card already full, not adding");

        return -1;

    }

    ret = ccid_card_initfn(card);

    if (ret == 0) {

        s->card = card;

    }

    return ret;

}

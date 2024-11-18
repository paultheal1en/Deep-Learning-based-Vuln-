static void uhci_async_cancel_device(UHCIState *s, USBDevice *dev)

{

    UHCIAsync *curr, *n;



    QTAILQ_FOREACH_SAFE(curr, &s->async_pending, next, n) {

        if (curr->packet.owner == NULL ||

            curr->packet.owner->dev != dev) {

            continue;

        }

        uhci_async_unlink(s, curr);

        uhci_async_cancel(s, curr);

    }

}

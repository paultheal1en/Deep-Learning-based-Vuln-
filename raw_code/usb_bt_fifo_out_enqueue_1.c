static inline void usb_bt_fifo_out_enqueue(struct USBBtState *s,

                struct usb_hci_out_fifo_s *fifo,

                void (*send)(struct HCIInfo *, const uint8_t *, int),

                int (*complete)(const uint8_t *, int),

                const uint8_t *data, int len)

{

    if (fifo->len) {

        memcpy(fifo->data + fifo->len, data, len);

        fifo->len += len;

        if (complete(fifo->data, fifo->len)) {

            send(s->hci, fifo->data, fifo->len);

            fifo->len = 0;

        }

    } else if (complete(data, len))

        send(s->hci, data, len);

    else {

        memcpy(fifo->data, data, len);

        fifo->len = len;

    }



    /* TODO: do we need to loop? */

}

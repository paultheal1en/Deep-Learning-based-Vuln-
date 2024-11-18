static void async_complete(void *opaque)

{

    USBHostDevice *s = opaque;

    AsyncURB *aurb;

    int urbs = 0;



    while (1) {

        USBPacket *p;



        int r = ioctl(s->fd, USBDEVFS_REAPURBNDELAY, &aurb);

        if (r < 0) {

            if (errno == EAGAIN) {

                if (urbs > 2) {

                    fprintf(stderr, "husb: %d iso urbs finished at once\n", urbs);

                }

                return;

            }

            if (errno == ENODEV && !s->closing) {

                do_disconnect(s);

                return;

            }



            DPRINTF("husb: async. reap urb failed errno %d\n", errno);

            return;

        }



        DPRINTF("husb: async completed. aurb %p status %d alen %d\n",

                aurb, aurb->urb.status, aurb->urb.actual_length);



        /* If this is a buffered iso urb mark it as complete and don't do

           anything else (it is handled further in usb_host_handle_iso_data) */

        if (aurb->iso_frame_idx == -1) {

            int inflight;

            if (aurb->urb.status == -EPIPE) {

                set_halt(s, aurb->urb.endpoint & 0xf);

            }

            aurb->iso_frame_idx = 0;

            urbs++;

            inflight = change_iso_inflight(s, aurb->urb.endpoint & 0xf, -1);

            if (inflight == 0 && is_iso_started(s, aurb->urb.endpoint & 0xf)) {

                fprintf(stderr, "husb: out of buffers for iso stream\n");

            }

            continue;

        }



        p = aurb->packet;



        if (p) {

            switch (aurb->urb.status) {

            case 0:

                p->len += aurb->urb.actual_length;

                break;



            case -EPIPE:

                set_halt(s, p->devep);

                p->len = USB_RET_STALL;

                break;



            default:

                p->len = USB_RET_NAK;

                break;

            }



            if (aurb->urb.type == USBDEVFS_URB_TYPE_CONTROL) {

                usb_generic_async_ctrl_complete(&s->dev, p);

            } else if (!aurb->more) {

                usb_packet_complete(&s->dev, p);

            }

        }



        async_free(aurb);

    }

}

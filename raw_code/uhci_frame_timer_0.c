static void uhci_frame_timer(void *opaque)

{

    UHCIState *s = opaque;

    int64_t expire_time;

    uint32_t frame_addr, link, old_td_ctrl, val, int_mask;

    int cnt, ret;

    UHCI_TD td;

    UHCI_QH qh;

    uint32_t old_async_qh;



    if (!(s->cmd & UHCI_CMD_RS)) {

        qemu_del_timer(s->frame_timer);

        /* set hchalted bit in status - UHCI11D 2.1.2 */

        s->status |= UHCI_STS_HCHALTED;

        return;

    }

    /* Complete the previous frame.  */

    s->frnum = (s->frnum + 1) & 0x7ff;

    if (s->pending_int_mask) {

        s->status2 |= s->pending_int_mask;

        s->status |= UHCI_STS_USBINT;

        uhci_update_irq(s);

    }

    old_async_qh = s->async_qh;

    frame_addr = s->fl_base_addr + ((s->frnum & 0x3ff) << 2);

    cpu_physical_memory_read(frame_addr, (uint8_t *)&link, 4);

    le32_to_cpus(&link);

    int_mask = 0;

    cnt = FRAME_MAX_LOOPS;

    while ((link & 1) == 0) {

        if (--cnt == 0)

            break;

        /* valid frame */

        if (link & 2) {

            /* QH */

            if (link == s->async_qh) {

                /* We've found a previously issues packet.

                   Nothing else to do.  */

                old_async_qh = 0;

                break;

            }

            cpu_physical_memory_read(link & ~0xf, (uint8_t *)&qh, sizeof(qh));

            le32_to_cpus(&qh.link);

            le32_to_cpus(&qh.el_link);

        depth_first:

            if (qh.el_link & 1) {

                /* no element : go to next entry */

                link = qh.link;

            } else if (qh.el_link & 2) {

                /* QH */

                link = qh.el_link;

            } else if (s->async_qh) {

                /* We can only cope with one pending packet.  Keep looking

                   for the previously issued packet.  */

                link = qh.link;

            } else {

                /* TD */

                if (--cnt == 0)

                    break;

                cpu_physical_memory_read(qh.el_link & ~0xf,

                                         (uint8_t *)&td, sizeof(td));

                le32_to_cpus(&td.link);

                le32_to_cpus(&td.ctrl);

                le32_to_cpus(&td.token);

                le32_to_cpus(&td.buffer);

                old_td_ctrl = td.ctrl;

                ret = uhci_handle_td(s, &td, &int_mask, 0);



                /* update the status bits of the TD */

                if (old_td_ctrl != td.ctrl) {

                    val = cpu_to_le32(td.ctrl);

                    cpu_physical_memory_write((qh.el_link & ~0xf) + 4,

                                              (const uint8_t *)&val,

                                              sizeof(val));

                }

                if (ret < 0)

                    break; /* interrupted frame */

                if (ret == 2) {

                    s->async_qh = link;

                } else if (ret == 0) {

                    /* update qh element link */

                    qh.el_link = td.link;

                    val = cpu_to_le32(qh.el_link);

                    cpu_physical_memory_write((link & ~0xf) + 4,

                                              (const uint8_t *)&val,

                                              sizeof(val));

                    if (qh.el_link & 4) {

                        /* depth first */

                        goto depth_first;

                    }

                }

                /* go to next entry */

                link = qh.link;

            }

        } else {

            /* TD */

            cpu_physical_memory_read(link & ~0xf, (uint8_t *)&td, sizeof(td));

            le32_to_cpus(&td.link);

            le32_to_cpus(&td.ctrl);

            le32_to_cpus(&td.token);

            le32_to_cpus(&td.buffer);



            /* Handle isochonous transfer.  */

            /* FIXME: might be more than one isoc in frame */

            old_td_ctrl = td.ctrl;

            ret = uhci_handle_td(s, &td, &int_mask, 0);



            /* update the status bits of the TD */

            if (old_td_ctrl != td.ctrl) {

                val = cpu_to_le32(td.ctrl);

                cpu_physical_memory_write((link & ~0xf) + 4,

                                          (const uint8_t *)&val,

                                          sizeof(val));

            }

            if (ret < 0)

                break; /* interrupted frame */

            if (ret == 2) {

                s->async_frame_addr = frame_addr;

            }

            link = td.link;

        }

    }

    s->pending_int_mask = int_mask;

    if (old_async_qh) {

        /* A previously started transfer has disappeared from the transfer

           list.  There's nothing useful we can do with it now, so just

           discard the packet and hope it wasn't too important.  */

#ifdef DEBUG

        printf("Discarding USB packet\n");

#endif

        usb_cancel_packet(&s->usb_packet);

        s->async_qh = 0;

    }



    /* prepare the timer for the next frame */

    expire_time = qemu_get_clock(vm_clock) +

        (ticks_per_sec / FRAME_TIMER_FREQ);

    qemu_mod_timer(s->frame_timer, expire_time);

}

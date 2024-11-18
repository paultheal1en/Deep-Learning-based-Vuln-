static void uhci_async_complete_packet(USBPacket * packet, void *opaque)

{

    UHCIState *s = opaque;

    UHCI_QH qh;

    UHCI_TD td;

    uint32_t link;

    uint32_t old_td_ctrl;

    uint32_t val;

    uint32_t frame_addr;

    int ret;



    /* Handle async isochronous packet completion */

    frame_addr = s->async_frame_addr;

    if (frame_addr) {

        cpu_physical_memory_read(frame_addr, (uint8_t *)&link, 4);

        le32_to_cpus(&link);



        cpu_physical_memory_read(link & ~0xf, (uint8_t *)&td, sizeof(td));

        le32_to_cpus(&td.link);

        le32_to_cpus(&td.ctrl);

        le32_to_cpus(&td.token);

        le32_to_cpus(&td.buffer);

        old_td_ctrl = td.ctrl;

        ret = uhci_handle_td(s, &td, &s->pending_int_mask, 1);



        /* update the status bits of the TD */

        if (old_td_ctrl != td.ctrl) {

            val = cpu_to_le32(td.ctrl);

            cpu_physical_memory_write((link & ~0xf) + 4,

                                      (const uint8_t *)&val,

                                      sizeof(val));

        }

        if (ret == 2) {

            s->async_frame_addr = frame_addr;

        } else if (ret == 0) {

            /* update qh element link */

            val = cpu_to_le32(td.link);

            cpu_physical_memory_write(frame_addr,

                                      (const uint8_t *)&val,

                                      sizeof(val));

        }

        return;

    }



    link = s->async_qh;

    if (!link) {

        /* This should never happen. It means a TD somehow got removed

           without cancelling the associated async IO request.  */

        return;

    }

    cpu_physical_memory_read(link & ~0xf, (uint8_t *)&qh, sizeof(qh));

    le32_to_cpus(&qh.link);

    le32_to_cpus(&qh.el_link);

    /* Re-process the queue containing the async packet.  */

    while (1) {

        cpu_physical_memory_read(qh.el_link & ~0xf,

                                 (uint8_t *)&td, sizeof(td));

        le32_to_cpus(&td.link);

        le32_to_cpus(&td.ctrl);

        le32_to_cpus(&td.token);

        le32_to_cpus(&td.buffer);

        old_td_ctrl = td.ctrl;

        ret = uhci_handle_td(s, &td, &s->pending_int_mask, 1);



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

            break;

        } else if (ret == 0) {

            /* update qh element link */

            qh.el_link = td.link;

            val = cpu_to_le32(qh.el_link);

            cpu_physical_memory_write((link & ~0xf) + 4,

                                      (const uint8_t *)&val,

                                      sizeof(val));

            if (!(qh.el_link & 4))

                break;

        }

        break;

    }

}

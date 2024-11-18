static void uhci_process_frame(UHCIState *s)

{

    uint32_t frame_addr, link, old_td_ctrl, val, int_mask;

    uint32_t curr_qh, td_count = 0;

    int cnt, ret;

    UHCI_TD td;

    UHCI_QH qh;

    QhDb qhdb;



    frame_addr = s->fl_base_addr + ((s->frnum & 0x3ff) << 2);



    pci_dma_read(&s->dev, frame_addr, &link, 4);

    le32_to_cpus(&link);



    int_mask = 0;

    curr_qh  = 0;



    qhdb_reset(&qhdb);



    for (cnt = FRAME_MAX_LOOPS; is_valid(link) && cnt; cnt--) {

        if (s->frame_bytes >= s->frame_bandwidth) {

            /* We've reached the usb 1.1 bandwidth, which is

               1280 bytes/frame, stop processing */

            trace_usb_uhci_frame_stop_bandwidth();

            break;

        }

        if (is_qh(link)) {

            /* QH */

            trace_usb_uhci_qh_load(link & ~0xf);



            if (qhdb_insert(&qhdb, link)) {

                /*

                 * We're going in circles. Which is not a bug because

                 * HCD is allowed to do that as part of the BW management.

                 *

                 * Stop processing here if no transaction has been done

                 * since we've been here last time.

                 */

                if (td_count == 0) {

                    trace_usb_uhci_frame_loop_stop_idle();

                    break;

                } else {

                    trace_usb_uhci_frame_loop_continue();

                    td_count = 0;

                    qhdb_reset(&qhdb);

                    qhdb_insert(&qhdb, link);

                }

            }



            pci_dma_read(&s->dev, link & ~0xf, &qh, sizeof(qh));

            le32_to_cpus(&qh.link);

            le32_to_cpus(&qh.el_link);



            if (!is_valid(qh.el_link)) {

                /* QH w/o elements */

                curr_qh = 0;

                link = qh.link;

            } else {

                /* QH with elements */

            	curr_qh = link;

            	link = qh.el_link;

            }

            continue;

        }



        /* TD */

        uhci_read_td(s, &td, link);

        trace_usb_uhci_td_load(curr_qh & ~0xf, link & ~0xf, td.ctrl, td.token);



        old_td_ctrl = td.ctrl;

        ret = uhci_handle_td(s, NULL, &td, link, &int_mask);

        if (old_td_ctrl != td.ctrl) {

            /* update the status bits of the TD */

            val = cpu_to_le32(td.ctrl);

            pci_dma_write(&s->dev, (link & ~0xf) + 4, &val, sizeof(val));

        }



        switch (ret) {

        case TD_RESULT_STOP_FRAME: /* interrupted frame */

            goto out;



        case TD_RESULT_NEXT_QH:

        case TD_RESULT_ASYNC_CONT:

            trace_usb_uhci_td_nextqh(curr_qh & ~0xf, link & ~0xf);

            link = curr_qh ? qh.link : td.link;

            continue;



        case TD_RESULT_ASYNC_START:

            trace_usb_uhci_td_async(curr_qh & ~0xf, link & ~0xf);

            link = curr_qh ? qh.link : td.link;

            continue;



        case TD_RESULT_COMPLETE:

            trace_usb_uhci_td_complete(curr_qh & ~0xf, link & ~0xf);

            link = td.link;

            td_count++;

            s->frame_bytes += (td.ctrl & 0x7ff) + 1;



            if (curr_qh) {

                /* update QH element link */

                qh.el_link = link;

                val = cpu_to_le32(qh.el_link);

                pci_dma_write(&s->dev, (curr_qh & ~0xf) + 4, &val, sizeof(val));



                if (!depth_first(link)) {

                    /* done with this QH */

                    curr_qh = 0;

                    link    = qh.link;

                }

            }

            break;



        default:

            assert(!"unknown return code");

        }



        /* go to the next entry */

    }



out:

    s->pending_int_mask |= int_mask;

}

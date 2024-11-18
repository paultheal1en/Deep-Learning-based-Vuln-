static void uhci_process_frame(UHCIState *s)

{

    uint32_t frame_addr, link, old_td_ctrl, val, int_mask;

    uint32_t curr_qh;

    int cnt, ret;

    UHCI_TD td;

    UHCI_QH qh;

    QhDb qhdb;



    frame_addr = s->fl_base_addr + ((s->frnum & 0x3ff) << 2);



    DPRINTF("uhci: processing frame %d addr 0x%x\n" , s->frnum, frame_addr);



    cpu_physical_memory_read(frame_addr, (uint8_t *)&link, 4);

    le32_to_cpus(&link);



    int_mask = 0;

    curr_qh  = 0;



    qhdb_reset(&qhdb);



    for (cnt = FRAME_MAX_LOOPS; is_valid(link) && cnt; cnt--) {

        if (is_qh(link)) {

            /* QH */



            if (qhdb_insert(&qhdb, link)) {

                /*

                 * We're going in circles. Which is not a bug because

                 * HCD is allowed to do that as part of the BW management. 

                 * In our case though it makes no sense to spin here. Sync transations 

                 * are already done, and async completion handler will re-process 

                 * the frame when something is ready.

                 */

                DPRINTF("uhci: detected loop. qh 0x%x\n", link);

                break;

            }



            cpu_physical_memory_read(link & ~0xf, (uint8_t *) &qh, sizeof(qh));

            le32_to_cpus(&qh.link);

            le32_to_cpus(&qh.el_link);



            DPRINTF("uhci: QH 0x%x load. link 0x%x elink 0x%x\n",

                    link, qh.link, qh.el_link);



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

        cpu_physical_memory_read(link & ~0xf, (uint8_t *) &td, sizeof(td));

        le32_to_cpus(&td.link);

        le32_to_cpus(&td.ctrl);

        le32_to_cpus(&td.token);

        le32_to_cpus(&td.buffer);



        DPRINTF("uhci: TD 0x%x load. link 0x%x ctrl 0x%x token 0x%x qh 0x%x\n", 

                link, td.link, td.ctrl, td.token, curr_qh);



        old_td_ctrl = td.ctrl;

        ret = uhci_handle_td(s, link, &td, &int_mask);

        if (old_td_ctrl != td.ctrl) {

            /* update the status bits of the TD */

            val = cpu_to_le32(td.ctrl);

            cpu_physical_memory_write((link & ~0xf) + 4,

                                      (const uint8_t *)&val, sizeof(val));

        }



        if (ret < 0) {

            /* interrupted frame */

            break;

        }



        if (ret == 2 || ret == 1) {

            DPRINTF("uhci: TD 0x%x %s. link 0x%x ctrl 0x%x token 0x%x qh 0x%x\n",

                    link, ret == 2 ? "pend" : "skip",

                    td.link, td.ctrl, td.token, curr_qh);



            link = curr_qh ? qh.link : td.link;

            continue;

        }



        /* completed TD */



        DPRINTF("uhci: TD 0x%x done. link 0x%x ctrl 0x%x token 0x%x qh 0x%x\n", 

                link, td.link, td.ctrl, td.token, curr_qh);



        link = td.link;



        if (curr_qh) {

	    /* update QH element link */

            qh.el_link = link;

            val = cpu_to_le32(qh.el_link);

            cpu_physical_memory_write((curr_qh & ~0xf) + 4,

                                          (const uint8_t *)&val, sizeof(val));



            if (!depth_first(link)) {

               /* done with this QH */



               DPRINTF("uhci: QH 0x%x done. link 0x%x elink 0x%x\n",

                       curr_qh, qh.link, qh.el_link);



               curr_qh = 0;

               link    = qh.link;

            }

        }



        /* go to the next entry */

    }



    s->pending_int_mask |= int_mask;

}

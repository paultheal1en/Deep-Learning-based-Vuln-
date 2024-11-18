static void uhci_fill_queue(UHCIState *s, UHCI_TD *td)

{

    uint32_t int_mask = 0;

    uint32_t plink = td->link;

    uint32_t token = uhci_queue_token(td);

    UHCI_TD ptd;

    int ret;



    while (is_valid(plink)) {

        pci_dma_read(&s->dev, plink & ~0xf, &ptd, sizeof(ptd));

        le32_to_cpus(&ptd.link);

        le32_to_cpus(&ptd.ctrl);

        le32_to_cpus(&ptd.token);

        le32_to_cpus(&ptd.buffer);

        if (!(ptd.ctrl & TD_CTRL_ACTIVE)) {

            break;

        }

        if (uhci_queue_token(&ptd) != token) {

            break;

        }

        trace_usb_uhci_td_queue(plink & ~0xf, ptd.ctrl, ptd.token);

        ret = uhci_handle_td(s, plink, &ptd, &int_mask, true);

        if (ret == TD_RESULT_ASYNC_CONT) {

            break;

        }

        assert(ret == TD_RESULT_ASYNC_START);

        assert(int_mask == 0);

        if (ptd.ctrl & TD_CTRL_SPD) {

            break;

        }

        plink = ptd.link;

    }

}

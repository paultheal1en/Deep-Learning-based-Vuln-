static inline void sdhci_blkgap_write(SDHCIState *s, uint8_t value)

{

    if ((value & SDHC_STOP_AT_GAP_REQ) && (s->blkgap & SDHC_STOP_AT_GAP_REQ)) {

        return;

    }

    s->blkgap = value & SDHC_STOP_AT_GAP_REQ;



    if ((value & SDHC_CONTINUE_REQ) && s->stopped_state &&

            (s->blkgap & SDHC_STOP_AT_GAP_REQ) == 0) {

        if (s->stopped_state == sdhc_gap_read) {

            s->prnsts |= SDHC_DAT_LINE_ACTIVE | SDHC_DOING_READ;

            SDHCI_GET_CLASS(s)->read_block_from_card(s);

        } else {

            s->prnsts |= SDHC_DAT_LINE_ACTIVE | SDHC_DOING_WRITE;

            SDHCI_GET_CLASS(s)->write_block_to_card(s);

        }

        s->stopped_state = sdhc_not_stopped;

    } else if (!s->stopped_state && (value & SDHC_STOP_AT_GAP_REQ)) {

        if (s->prnsts & SDHC_DOING_READ) {

            s->stopped_state = sdhc_gap_read;

        } else if (s->prnsts & SDHC_DOING_WRITE) {

            s->stopped_state = sdhc_gap_write;

        }

    }

}

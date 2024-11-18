static inline void sdhci_reset_write(SDHCIState *s, uint8_t value)

{

    switch (value) {

    case SDHC_RESET_ALL:

        DEVICE_GET_CLASS(s)->reset(DEVICE(s));

        break;

    case SDHC_RESET_CMD:

        s->prnsts &= ~SDHC_CMD_INHIBIT;

        s->norintsts &= ~SDHC_NIS_CMDCMP;

        break;

    case SDHC_RESET_DATA:

        s->data_count = 0;

        s->prnsts &= ~(SDHC_SPACE_AVAILABLE | SDHC_DATA_AVAILABLE |

                SDHC_DOING_READ | SDHC_DOING_WRITE |

                SDHC_DATA_INHIBIT | SDHC_DAT_LINE_ACTIVE);

        s->blkgap &= ~(SDHC_STOP_AT_GAP_REQ | SDHC_CONTINUE_REQ);

        s->stopped_state = sdhc_not_stopped;

        s->norintsts &= ~(SDHC_NIS_WBUFRDY | SDHC_NIS_RBUFRDY |

                SDHC_NIS_DMA | SDHC_NIS_TRSCMP | SDHC_NIS_BLKGAP);

        break;

    }

}

static void sdhci_data_transfer(SDHCIState *s)

{

    SDHCIClass *k = SDHCI_GET_CLASS(s);



    if (s->trnmod & SDHC_TRNS_DMA) {

        switch (SDHC_DMA_TYPE(s->hostctl)) {

        case SDHC_CTRL_SDMA:

            if ((s->trnmod & SDHC_TRNS_MULTI) &&

                    (!(s->trnmod & SDHC_TRNS_BLK_CNT_EN) || s->blkcnt == 0)) {

                break;

            }



            if ((s->blkcnt == 1) || !(s->trnmod & SDHC_TRNS_MULTI)) {

                k->do_sdma_single(s);

            } else {

                k->do_sdma_multi(s);

            }



            break;

        case SDHC_CTRL_ADMA1_32:

            if (!(s->capareg & SDHC_CAN_DO_ADMA1)) {

                ERRPRINT("ADMA1 not supported\n");

                break;

            }



            k->do_adma(s);

            break;

        case SDHC_CTRL_ADMA2_32:

            if (!(s->capareg & SDHC_CAN_DO_ADMA2)) {

                ERRPRINT("ADMA2 not supported\n");

                break;

            }



            k->do_adma(s);

            break;

        case SDHC_CTRL_ADMA2_64:

            if (!(s->capareg & SDHC_CAN_DO_ADMA2) ||

                    !(s->capareg & SDHC_64_BIT_BUS_SUPPORT)) {

                ERRPRINT("64 bit ADMA not supported\n");

                break;

            }



            k->do_adma(s);

            break;

        default:

            ERRPRINT("Unsupported DMA type\n");

            break;

        }

    } else {

        if ((s->trnmod & SDHC_TRNS_READ) && sd_data_ready(s->card)) {

            s->prnsts |= SDHC_DOING_READ | SDHC_DATA_INHIBIT |

                    SDHC_DAT_LINE_ACTIVE;

            SDHCI_GET_CLASS(s)->read_block_from_card(s);

        } else {

            s->prnsts |= SDHC_DOING_WRITE | SDHC_DAT_LINE_ACTIVE |

                    SDHC_SPACE_AVAILABLE | SDHC_DATA_INHIBIT;

            SDHCI_GET_CLASS(s)->write_block_to_card(s);

        }

    }

}

static void sdhci_data_transfer(void *opaque)

{

    SDHCIState *s = (SDHCIState *)opaque;



    if (s->trnmod & SDHC_TRNS_DMA) {

        switch (SDHC_DMA_TYPE(s->hostctl)) {

        case SDHC_CTRL_SDMA:

            if ((s->blkcnt == 1) || !(s->trnmod & SDHC_TRNS_MULTI)) {

                sdhci_sdma_transfer_single_block(s);

            } else {

                sdhci_sdma_transfer_multi_blocks(s);

            }



            break;

        case SDHC_CTRL_ADMA1_32:

            if (!(s->capareg & SDHC_CAN_DO_ADMA1)) {

                ERRPRINT("ADMA1 not supported\n");

                break;

            }



            sdhci_do_adma(s);

            break;

        case SDHC_CTRL_ADMA2_32:

            if (!(s->capareg & SDHC_CAN_DO_ADMA2)) {

                ERRPRINT("ADMA2 not supported\n");

                break;

            }



            sdhci_do_adma(s);

            break;

        case SDHC_CTRL_ADMA2_64:

            if (!(s->capareg & SDHC_CAN_DO_ADMA2) ||

                    !(s->capareg & SDHC_64_BIT_BUS_SUPPORT)) {

                ERRPRINT("64 bit ADMA not supported\n");

                break;

            }



            sdhci_do_adma(s);

            break;

        default:

            ERRPRINT("Unsupported DMA type\n");

            break;

        }

    } else {

        if ((s->trnmod & SDHC_TRNS_READ) && sdbus_data_ready(&s->sdbus)) {

            s->prnsts |= SDHC_DOING_READ | SDHC_DATA_INHIBIT |

                    SDHC_DAT_LINE_ACTIVE;

            sdhci_read_block_from_card(s);

        } else {

            s->prnsts |= SDHC_DOING_WRITE | SDHC_DAT_LINE_ACTIVE |

                    SDHC_SPACE_AVAILABLE | SDHC_DATA_INHIBIT;

            sdhci_write_block_to_card(s);

        }

    }

}

static void ccid_card_vscard_handle_message(PassthruState *card,

    VSCMsgHeader *scr_msg_header)

{

    uint8_t *data = (uint8_t *)&scr_msg_header[1];



    switch (scr_msg_header->type) {

    case VSC_ATR:

        DPRINTF(card, D_INFO, "VSC_ATR %d\n", scr_msg_header->length);

        if (scr_msg_header->length > MAX_ATR_SIZE) {

            error_report("ATR size exceeds spec, ignoring");

            ccid_card_vscard_send_error(card, scr_msg_header->reader_id,

                                        VSC_GENERAL_ERROR);


        }

        memcpy(card->atr, data, scr_msg_header->length);

        card->atr_length = scr_msg_header->length;

        ccid_card_card_inserted(&card->base);

        ccid_card_vscard_send_error(card, scr_msg_header->reader_id,

                                    VSC_SUCCESS);


    case VSC_APDU:

        ccid_card_send_apdu_to_guest(

            &card->base, data, scr_msg_header->length);


    case VSC_CardRemove:

        DPRINTF(card, D_INFO, "VSC_CardRemove\n");

        ccid_card_card_removed(&card->base);

        ccid_card_vscard_send_error(card,

            scr_msg_header->reader_id, VSC_SUCCESS);


    case VSC_Init:

        ccid_card_vscard_handle_init(

            card, scr_msg_header, (VSCMsgInit *)data);


    case VSC_Error:

        ccid_card_card_error(&card->base, *(uint32_t *)data);


    case VSC_ReaderAdd:

        if (ccid_card_ccid_attach(&card->base) < 0) {

            ccid_card_vscard_send_error(card, VSCARD_UNDEFINED_READER_ID,

                                      VSC_CANNOT_ADD_MORE_READERS);

        } else {

            ccid_card_vscard_send_error(card, VSCARD_MINIMAL_READER_ID,

                                        VSC_SUCCESS);

        }


    case VSC_ReaderRemove:

        ccid_card_ccid_detach(&card->base);

        ccid_card_vscard_send_error(card,

            scr_msg_header->reader_id, VSC_SUCCESS);


    default:

        printf("usb-ccid: chardev: unexpected message of type %X\n",

               scr_msg_header->type);

        ccid_card_vscard_send_error(card, scr_msg_header->reader_id,

            VSC_GENERAL_ERROR);

    }

}
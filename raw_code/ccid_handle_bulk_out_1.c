static int ccid_handle_bulk_out(USBCCIDState *s, USBPacket *p)

{

    CCID_Header *ccid_header;



    if (p->len + s->bulk_out_pos > BULK_OUT_DATA_SIZE) {

        return USB_RET_STALL;

    }

    ccid_header = (CCID_Header *)s->bulk_out_data;

    memcpy(s->bulk_out_data + s->bulk_out_pos, p->data, p->len);

    s->bulk_out_pos += p->len;

    if (p->len == CCID_MAX_PACKET_SIZE) {

        DPRINTF(s, D_VERBOSE,

            "usb-ccid: bulk_in: expecting more packets (%d/%d)\n",

            p->len, ccid_header->dwLength);

        return 0;

    }

    if (s->bulk_out_pos < 10) {

        DPRINTF(s, 1,

                "%s: bad USB_TOKEN_OUT length, should be at least 10 bytes\n",

                __func__);

    } else {

        DPRINTF(s, D_MORE_INFO, "%s %x\n", __func__, ccid_header->bMessageType);

        switch (ccid_header->bMessageType) {

        case CCID_MESSAGE_TYPE_PC_to_RDR_GetSlotStatus:

            ccid_write_slot_status(s, ccid_header);

            break;

        case CCID_MESSAGE_TYPE_PC_to_RDR_IccPowerOn:

            DPRINTF(s, 1, "PowerOn: %d\n",

                ((CCID_IccPowerOn *)(ccid_header))->bPowerSelect);

            s->powered = true;

            if (!ccid_card_inserted(s)) {

                ccid_report_error_failed(s, ERROR_ICC_MUTE);

            }

            /* atr is written regardless of error. */

            ccid_write_data_block_atr(s, ccid_header);

            break;

        case CCID_MESSAGE_TYPE_PC_to_RDR_IccPowerOff:

            DPRINTF(s, 1, "PowerOff\n");

            ccid_reset_error_status(s);

            s->powered = false;

            ccid_write_slot_status(s, ccid_header);

            break;

        case CCID_MESSAGE_TYPE_PC_to_RDR_XfrBlock:

            ccid_on_apdu_from_guest(s, (CCID_XferBlock *)s->bulk_out_data);

            break;

        case CCID_MESSAGE_TYPE_PC_to_RDR_SetParameters:

            ccid_reset_error_status(s);

            ccid_set_parameters(s, ccid_header);

            ccid_write_parameters(s, ccid_header);

            break;

        case CCID_MESSAGE_TYPE_PC_to_RDR_ResetParameters:

            ccid_reset_error_status(s);

            ccid_reset_parameters(s);

            ccid_write_parameters(s, ccid_header);

            break;

        case CCID_MESSAGE_TYPE_PC_to_RDR_GetParameters:

            ccid_reset_error_status(s);

            ccid_write_parameters(s, ccid_header);

            break;

        default:

            DPRINTF(s, 1,

                "handle_data: ERROR: unhandled message type %Xh\n",

                ccid_header->bMessageType);

            /*

             * The caller is expecting the device to respond, tell it we

             * don't support the operation.

             */

            ccid_report_error_failed(s, ERROR_CMD_NOT_SUPPORTED);

            ccid_write_slot_status(s, ccid_header);

            break;

        }

    }

    s->bulk_out_pos = 0;

    return 0;

}

cac_applet_pki_process_apdu(VCard *card, VCardAPDU *apdu,

                            VCardResponse **response)

{

    CACPKIAppletData *pki_applet = NULL;

    VCardAppletPrivate *applet_private = NULL;

    int size, next;

    unsigned char *sign_buffer;

    vcard_7816_status_t status;

    VCardStatus ret = VCARD_FAIL;



    applet_private = vcard_get_current_applet_private(card, apdu->a_channel);

    assert(applet_private);

    pki_applet = &(applet_private->u.pki_data);



    switch (apdu->a_ins) {

    case CAC_UPDATE_BUFFER:

        *response = vcard_make_response(

            VCARD7816_STATUS_ERROR_CONDITION_NOT_SATISFIED);

        ret = VCARD_DONE;

        break;

    case CAC_GET_CERTIFICATE:

        if ((apdu->a_p2 != 0) || (apdu->a_p1 != 0)) {

            *response = vcard_make_response(

                             VCARD7816_STATUS_ERROR_P1_P2_INCORRECT);

            break;

        }

        assert(pki_applet->cert != NULL);

        size = apdu->a_Le;

        if (pki_applet->cert_buffer == NULL) {

            pki_applet->cert_buffer = pki_applet->cert;

            pki_applet->cert_buffer_len = pki_applet->cert_len;

        }

        size = MIN(size, pki_applet->cert_buffer_len);

        next = MIN(255, pki_applet->cert_buffer_len - size);

        *response = vcard_response_new_bytes(

                        card, pki_applet->cert_buffer, size,

                        apdu->a_Le, next ?

                        VCARD7816_SW1_WARNING_CHANGE :

                        VCARD7816_SW1_SUCCESS,

                        next);

        pki_applet->cert_buffer += size;

        pki_applet->cert_buffer_len -= size;

        if ((*response == NULL) || (next == 0)) {

            pki_applet->cert_buffer = NULL;

        }

        if (*response == NULL) {

            *response = vcard_make_response(

                            VCARD7816_STATUS_EXC_ERROR_MEMORY_FAILURE);

        }

        ret = VCARD_DONE;

        break;

    case CAC_SIGN_DECRYPT:

        if (apdu->a_p2 != 0) {

            *response = vcard_make_response(

                             VCARD7816_STATUS_ERROR_P1_P2_INCORRECT);

            break;

        }

        size = apdu->a_Lc;



        sign_buffer = g_realloc(pki_applet->sign_buffer,

                                pki_applet->sign_buffer_len + size);

        memcpy(sign_buffer+pki_applet->sign_buffer_len, apdu->a_body, size);

        size += pki_applet->sign_buffer_len;

        switch (apdu->a_p1) {

        case  0x80:

            /* p1 == 0x80 means we haven't yet sent the whole buffer, wait for

             * the rest */

            pki_applet->sign_buffer = sign_buffer;

            pki_applet->sign_buffer_len = size;

            *response = vcard_make_response(VCARD7816_STATUS_SUCCESS);

            break;

        case 0x00:

            /* we now have the whole buffer, do the operation, result will be

             * in the sign_buffer */

            status = vcard_emul_rsa_op(card, pki_applet->key,

                                       sign_buffer, size);

            if (status != VCARD7816_STATUS_SUCCESS) {

                *response = vcard_make_response(status);

                break;

            }

            *response = vcard_response_new(card, sign_buffer, size, apdu->a_Le,

                                                     VCARD7816_STATUS_SUCCESS);

            if (*response == NULL) {

                *response = vcard_make_response(

                                VCARD7816_STATUS_EXC_ERROR_MEMORY_FAILURE);

            }

            break;

        default:

           *response = vcard_make_response(

                                VCARD7816_STATUS_ERROR_P1_P2_INCORRECT);

            break;

        }

        g_free(sign_buffer);

        pki_applet->sign_buffer = NULL;

        pki_applet->sign_buffer_len = 0;

        ret = VCARD_DONE;

        break;

    case CAC_READ_BUFFER:

        /* new CAC call, go ahead and use the old version for now */

        /* TODO: implement */

        *response = vcard_make_response(

                                VCARD7816_STATUS_ERROR_COMMAND_NOT_SUPPORTED);

        ret = VCARD_DONE;

        break;

    default:

        ret = cac_common_process_apdu(card, apdu, response);

        break;

    }

    return ret;

}

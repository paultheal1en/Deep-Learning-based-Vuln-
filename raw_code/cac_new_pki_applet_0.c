cac_new_pki_applet(int i, const unsigned char *cert,

                   int cert_len, VCardKey *key)

{

    VCardAppletPrivate *applet_private = NULL;

    VCardApplet *applet = NULL;

    unsigned char pki_aid[] = { 0xa0, 0x00, 0x00, 0x00, 0x79, 0x01, 0x00 };

    int pki_aid_len = sizeof(pki_aid);



    pki_aid[pki_aid_len-1] = i;



    applet_private = cac_new_pki_applet_private(cert, cert_len, key);

    if (applet_private == NULL) {

        goto failure;

    }

    applet = vcard_new_applet(cac_applet_pki_process_apdu, cac_applet_pki_reset,

                              pki_aid, pki_aid_len);

    if (applet == NULL) {

        goto failure;

    }

    vcard_set_applet_private(applet, applet_private,

                             cac_delete_pki_applet_private);

    applet_private = NULL;



    return applet;



failure:

    if (applet_private != NULL) {

        cac_delete_pki_applet_private(applet_private);

    }

    return NULL;

}

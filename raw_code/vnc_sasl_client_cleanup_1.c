void vnc_sasl_client_cleanup(VncState *vs)

{

    if (vs->sasl.conn) {

        vs->sasl.runSSF = vs->sasl.waitWriteSSF = vs->sasl.wantSSF = 0;

        vs->sasl.encodedLength = vs->sasl.encodedOffset = 0;

        vs->sasl.encoded = NULL;

        g_free(vs->sasl.username);

        free(vs->sasl.mechlist);

        vs->sasl.username = vs->sasl.mechlist = NULL;

        sasl_dispose(&vs->sasl.conn);

        vs->sasl.conn = NULL;

    }

}

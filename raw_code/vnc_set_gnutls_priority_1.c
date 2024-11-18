static int vnc_set_gnutls_priority(gnutls_session_t s, int x509)

{

    const char *priority = x509 ? "NORMAL" : "NORMAL:+ANON-DH";

    int rc;



    rc = gnutls_priority_set_direct(s, priority, NULL);

    if (rc != GNUTLS_E_SUCCESS) {

        return -1;

    }

    return 0;

}

void test_tls_write_cert_chain(const char *filename,

                               gnutls_x509_crt_t *certs,

                               size_t ncerts)

{

    size_t i;

    size_t capacity = 1024, offset = 0;

    char *buffer = g_new0(char, capacity);

    int err;



    for (i = 0; i < ncerts; i++) {

        size_t len = capacity - offset;

    retry:

        err = gnutls_x509_crt_export(certs[i], GNUTLS_X509_FMT_PEM,

                                     buffer + offset, &len);

        if (err < 0) {

            if (err == GNUTLS_E_SHORT_MEMORY_BUFFER) {

                buffer = g_renew(char, buffer, offset + len);

                capacity = offset + len;

                goto retry;

            }

            g_critical("Failed to export certificate chain %s: %d",

                       gnutls_strerror(err), err);

            abort();

        }

        offset += len;

    }



    if (!g_file_set_contents(filename, buffer, offset, NULL)) {

        abort();

    }


}
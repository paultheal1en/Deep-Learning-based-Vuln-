static int vnc_set_x509_credential_dir(VncDisplay *vs,

				       const char *certdir)

{

    if (vnc_set_x509_credential(vs, certdir, X509_CA_CERT_FILE, &vs->x509cacert, 0) < 0)

	goto cleanup;

    if (vnc_set_x509_credential(vs, certdir, X509_CA_CRL_FILE, &vs->x509cacrl, 1) < 0)

	goto cleanup;

    if (vnc_set_x509_credential(vs, certdir, X509_SERVER_CERT_FILE, &vs->x509cert, 0) < 0)

	goto cleanup;

    if (vnc_set_x509_credential(vs, certdir, X509_SERVER_KEY_FILE, &vs->x509key, 0) < 0)

	goto cleanup;



    return 0;



 cleanup:

    qemu_free(vs->x509cacert);

    qemu_free(vs->x509cacrl);

    qemu_free(vs->x509cert);

    qemu_free(vs->x509key);

    vs->x509cacert = vs->x509cacrl = vs->x509cert = vs->x509key = NULL;

    return -1;

}

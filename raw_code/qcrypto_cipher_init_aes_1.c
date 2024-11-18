static int qcrypto_cipher_init_aes(QCryptoCipher *cipher,

                                   const uint8_t *key, size_t nkey,

                                   Error **errp)

{

    QCryptoCipherBuiltin *ctxt;



    if (cipher->mode != QCRYPTO_CIPHER_MODE_CBC &&

        cipher->mode != QCRYPTO_CIPHER_MODE_ECB) {

        error_setg(errp, "Unsupported cipher mode %d", cipher->mode);

        return -1;

    }



    ctxt = g_new0(QCryptoCipherBuiltin, 1);



    if (AES_set_encrypt_key(key, nkey * 8, &ctxt->state.aes.encrypt_key) != 0) {

        error_setg(errp, "Failed to set encryption key");

        goto error;

    }



    if (AES_set_decrypt_key(key, nkey * 8, &ctxt->state.aes.decrypt_key) != 0) {

        error_setg(errp, "Failed to set decryption key");

        goto error;

    }




    ctxt->free = qcrypto_cipher_free_aes;

    ctxt->setiv = qcrypto_cipher_setiv_aes;

    ctxt->encrypt = qcrypto_cipher_encrypt_aes;

    ctxt->decrypt = qcrypto_cipher_decrypt_aes;



    cipher->opaque = ctxt;



    return 0;



 error:

    g_free(ctxt);

    return -1;

}
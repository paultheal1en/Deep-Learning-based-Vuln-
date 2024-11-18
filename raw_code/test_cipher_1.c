static void test_cipher(const void *opaque)

{

    const QCryptoCipherTestData *data = opaque;



    QCryptoCipher *cipher;

    Error *err = NULL;

    uint8_t *key, *iv, *ciphertext, *plaintext, *outtext;

    size_t nkey, niv, nciphertext, nplaintext;

    char *outtexthex;



    g_test_message("foo");

    nkey = unhex_string(data->key, &key);

    niv = unhex_string(data->iv, &iv);

    nciphertext = unhex_string(data->ciphertext, &ciphertext);

    nplaintext = unhex_string(data->plaintext, &plaintext);



    g_assert(nciphertext == nplaintext);



    outtext = g_new0(uint8_t, nciphertext);



    cipher = qcrypto_cipher_new(

        data->alg, data->mode,

        key, nkey,

        &err);

    g_assert(cipher != NULL);

    g_assert(err == NULL);





    if (iv) {

        g_assert(qcrypto_cipher_setiv(cipher,

                                      iv, niv,

                                      &err) == 0);

        g_assert(err == NULL);

    }

    g_assert(qcrypto_cipher_encrypt(cipher,

                                    plaintext,

                                    outtext,

                                    nplaintext,

                                    &err) == 0);

    g_assert(err == NULL);



    outtexthex = hex_string(outtext, nciphertext);



    g_assert_cmpstr(outtexthex, ==, data->ciphertext);



    g_free(outtext);

    g_free(outtexthex);

    g_free(key);

    g_free(iv);

    g_free(ciphertext);

    g_free(plaintext);

    qcrypto_cipher_free(cipher);

}

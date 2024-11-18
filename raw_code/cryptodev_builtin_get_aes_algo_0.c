cryptodev_builtin_get_aes_algo(uint32_t key_len, Error **errp)

{

    int algo;



    if (key_len == 128 / 8) {

        algo = QCRYPTO_CIPHER_ALG_AES_128;

    } else if (key_len == 192 / 8) {

        algo = QCRYPTO_CIPHER_ALG_AES_192;

    } else if (key_len == 256 / 8) {

        algo = QCRYPTO_CIPHER_ALG_AES_256;

    } else {

        error_setg(errp, "Unsupported key length :%u", key_len);

        return -1;

    }



    return algo;

}

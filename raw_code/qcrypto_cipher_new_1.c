QCryptoCipher *qcrypto_cipher_new(QCryptoCipherAlgorithm alg,

                                  QCryptoCipherMode mode,

                                  const uint8_t *key, size_t nkey,

                                  Error **errp)

{

    QCryptoCipher *cipher;

    QCryptoCipherNettle *ctx;

    uint8_t *rfbkey;



    switch (mode) {

    case QCRYPTO_CIPHER_MODE_ECB:

    case QCRYPTO_CIPHER_MODE_CBC:

    case QCRYPTO_CIPHER_MODE_XTS:

    case QCRYPTO_CIPHER_MODE_CTR:

        break;

    default:

        error_setg(errp, "Unsupported cipher mode %s",

                   QCryptoCipherMode_lookup[mode]);

        return NULL;

    }



    if (!qcrypto_cipher_validate_key_length(alg, mode, nkey, errp)) {

        return NULL;

    }



    cipher = g_new0(QCryptoCipher, 1);

    cipher->alg = alg;

    cipher->mode = mode;



    ctx = g_new0(QCryptoCipherNettle, 1);



    switch (alg) {

    case QCRYPTO_CIPHER_ALG_DES_RFB:

        ctx->ctx = g_new0(struct des_ctx, 1);

        rfbkey = qcrypto_cipher_munge_des_rfb_key(key, nkey);

        des_set_key(ctx->ctx, rfbkey);

        g_free(rfbkey);



        ctx->alg_encrypt_native = des_encrypt_native;

        ctx->alg_decrypt_native = des_decrypt_native;

        ctx->alg_encrypt_wrapper = des_encrypt_wrapper;

        ctx->alg_decrypt_wrapper = des_decrypt_wrapper;



        ctx->blocksize = DES_BLOCK_SIZE;

        break;



    case QCRYPTO_CIPHER_ALG_AES_128:

    case QCRYPTO_CIPHER_ALG_AES_192:

    case QCRYPTO_CIPHER_ALG_AES_256:

        ctx->ctx = g_new0(QCryptoNettleAES, 1);



        if (mode == QCRYPTO_CIPHER_MODE_XTS) {

            ctx->ctx_tweak = g_new0(QCryptoNettleAES, 1);



            nkey /= 2;

            aes_set_encrypt_key(&((QCryptoNettleAES *)ctx->ctx)->enc,

                                nkey, key);

            aes_set_decrypt_key(&((QCryptoNettleAES *)ctx->ctx)->dec,

                                nkey, key);



            aes_set_encrypt_key(&((QCryptoNettleAES *)ctx->ctx_tweak)->enc,

                                nkey, key + nkey);

            aes_set_decrypt_key(&((QCryptoNettleAES *)ctx->ctx_tweak)->dec,

                                nkey, key + nkey);

        } else {

            aes_set_encrypt_key(&((QCryptoNettleAES *)ctx->ctx)->enc,

                                nkey, key);

            aes_set_decrypt_key(&((QCryptoNettleAES *)ctx->ctx)->dec,

                                nkey, key);

        }



        ctx->alg_encrypt_native = aes_encrypt_native;

        ctx->alg_decrypt_native = aes_decrypt_native;

        ctx->alg_encrypt_wrapper = aes_encrypt_wrapper;

        ctx->alg_decrypt_wrapper = aes_decrypt_wrapper;



        ctx->blocksize = AES_BLOCK_SIZE;

        break;



    case QCRYPTO_CIPHER_ALG_CAST5_128:

        ctx->ctx = g_new0(struct cast128_ctx, 1);



        if (mode == QCRYPTO_CIPHER_MODE_XTS) {

            ctx->ctx_tweak = g_new0(struct cast128_ctx, 1);



            nkey /= 2;

            cast5_set_key(ctx->ctx, nkey, key);

            cast5_set_key(ctx->ctx_tweak, nkey, key + nkey);

        } else {

            cast5_set_key(ctx->ctx, nkey, key);

        }



        ctx->alg_encrypt_native = cast128_encrypt_native;

        ctx->alg_decrypt_native = cast128_decrypt_native;

        ctx->alg_encrypt_wrapper = cast128_encrypt_wrapper;

        ctx->alg_decrypt_wrapper = cast128_decrypt_wrapper;



        ctx->blocksize = CAST128_BLOCK_SIZE;

        break;



    case QCRYPTO_CIPHER_ALG_SERPENT_128:

    case QCRYPTO_CIPHER_ALG_SERPENT_192:

    case QCRYPTO_CIPHER_ALG_SERPENT_256:

        ctx->ctx = g_new0(struct serpent_ctx, 1);



        if (mode == QCRYPTO_CIPHER_MODE_XTS) {

            ctx->ctx_tweak = g_new0(struct serpent_ctx, 1);



            nkey /= 2;

            serpent_set_key(ctx->ctx, nkey, key);

            serpent_set_key(ctx->ctx_tweak, nkey, key + nkey);

        } else {

            serpent_set_key(ctx->ctx, nkey, key);

        }



        ctx->alg_encrypt_native = serpent_encrypt_native;

        ctx->alg_decrypt_native = serpent_decrypt_native;

        ctx->alg_encrypt_wrapper = serpent_encrypt_wrapper;

        ctx->alg_decrypt_wrapper = serpent_decrypt_wrapper;



        ctx->blocksize = SERPENT_BLOCK_SIZE;

        break;



    case QCRYPTO_CIPHER_ALG_TWOFISH_128:

    case QCRYPTO_CIPHER_ALG_TWOFISH_192:

    case QCRYPTO_CIPHER_ALG_TWOFISH_256:

        ctx->ctx = g_new0(struct twofish_ctx, 1);



        if (mode == QCRYPTO_CIPHER_MODE_XTS) {

            ctx->ctx_tweak = g_new0(struct twofish_ctx, 1);



            nkey /= 2;

            twofish_set_key(ctx->ctx, nkey, key);

            twofish_set_key(ctx->ctx_tweak, nkey, key + nkey);

        } else {

            twofish_set_key(ctx->ctx, nkey, key);

        }



        ctx->alg_encrypt_native = twofish_encrypt_native;

        ctx->alg_decrypt_native = twofish_decrypt_native;

        ctx->alg_encrypt_wrapper = twofish_encrypt_wrapper;

        ctx->alg_decrypt_wrapper = twofish_decrypt_wrapper;



        ctx->blocksize = TWOFISH_BLOCK_SIZE;

        break;



    default:

        error_setg(errp, "Unsupported cipher algorithm %s",

                   QCryptoCipherAlgorithm_lookup[alg]);

        goto error;

    }



    if (mode == QCRYPTO_CIPHER_MODE_XTS &&

        ctx->blocksize != XTS_BLOCK_SIZE) {

        error_setg(errp, "Cipher block size %zu must equal XTS block size %d",

                   ctx->blocksize, XTS_BLOCK_SIZE);

        goto error;

    }



    ctx->iv = g_new0(uint8_t, ctx->blocksize);

    cipher->opaque = ctx;



    return cipher;



 error:

    g_free(cipher);

    g_free(ctx);

    return NULL;

}

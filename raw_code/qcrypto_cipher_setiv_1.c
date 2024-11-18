int qcrypto_cipher_setiv(QCryptoCipher *cipher,

                         const uint8_t *iv, size_t niv,

                         Error **errp)

{

    QCryptoCipherNettle *ctx = cipher->opaque;

    if (niv != ctx->niv) {

        error_setg(errp, "Expected IV size %zu not %zu",

                   ctx->niv, niv);

        return -1;

    }

    memcpy(ctx->iv, iv, niv);

    return 0;

}

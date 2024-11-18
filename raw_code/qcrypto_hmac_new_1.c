QCryptoHmac *qcrypto_hmac_new(QCryptoHashAlgorithm alg,

                              const uint8_t *key, size_t nkey,

                              Error **errp)

{

    QCryptoHmac *hmac;

    void *ctx = NULL;

    Error *err2 = NULL;

    QCryptoHmacDriver *drv = NULL;



#ifdef CONFIG_AF_ALG

    ctx = qcrypto_afalg_hmac_ctx_new(alg, key, nkey, &err2);

    if (ctx) {

        drv = &qcrypto_hmac_afalg_driver;

    }

#endif



    if (!ctx) {

        ctx = qcrypto_hmac_ctx_new(alg, key, nkey, errp);

        if (!ctx) {

            return NULL;

        }



        drv = &qcrypto_hmac_lib_driver;

        error_free(err2);

    }



    hmac = g_new0(QCryptoHmac, 1);

    hmac->alg = alg;

    hmac->opaque = ctx;

    hmac->driver = (void *)drv;



    return hmac;

}

static int qcrypto_ivgen_essiv_init(QCryptoIVGen *ivgen,

                                    const uint8_t *key, size_t nkey,

                                    Error **errp)

{

    uint8_t *salt;

    size_t nhash;

    size_t nsalt;

    QCryptoIVGenESSIV *essiv = g_new0(QCryptoIVGenESSIV, 1);



    /* Not necessarily the same as nkey */

    nsalt = qcrypto_cipher_get_key_len(ivgen->cipher);



    nhash = qcrypto_hash_digest_len(ivgen->hash);

    /* Salt must be larger of hash size or key size */

    salt = g_new0(uint8_t, MAX(nhash, nsalt));



    if (qcrypto_hash_bytes(ivgen->hash, (const gchar *)key, nkey,

                           &salt, &nhash,

                           errp) < 0) {

        g_free(essiv);


        return -1;

    }



    /* Now potentially truncate salt to match cipher key len */

    essiv->cipher = qcrypto_cipher_new(ivgen->cipher,

                                       QCRYPTO_CIPHER_MODE_ECB,

                                       salt, MIN(nhash, nsalt),

                                       errp);

    if (!essiv->cipher) {

        g_free(essiv);


        return -1;

    }




    ivgen->private = essiv;



    return 0;

}
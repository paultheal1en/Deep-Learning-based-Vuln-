static void qcow2_invalidate_cache(BlockDriverState *bs)
{
    BDRVQcowState *s = bs->opaque;
    int flags = s->flags;
    AES_KEY aes_encrypt_key;
    AES_KEY aes_decrypt_key;
    uint32_t crypt_method = 0;
    QDict *options;
    /*
     * Backing files are read-only which makes all of their metadata immutable,
     * that means we don't have to worry about reopening them here.
     */
    if (s->crypt_method) {
        crypt_method = s->crypt_method;
        memcpy(&aes_encrypt_key, &s->aes_encrypt_key, sizeof(aes_encrypt_key));
        memcpy(&aes_decrypt_key, &s->aes_decrypt_key, sizeof(aes_decrypt_key));
    }
    qcow2_close(bs);
    options = qdict_new();
    qdict_put(options, QCOW2_OPT_LAZY_REFCOUNTS,
              qbool_from_int(s->use_lazy_refcounts));
    memset(s, 0, sizeof(BDRVQcowState));
    qcow2_open(bs, options, flags, NULL);
    QDECREF(options);
    if (crypt_method) {
        s->crypt_method = crypt_method;
        memcpy(&s->aes_encrypt_key, &aes_encrypt_key, sizeof(aes_encrypt_key));
        memcpy(&s->aes_decrypt_key, &aes_decrypt_key, sizeof(aes_decrypt_key));
    }
}
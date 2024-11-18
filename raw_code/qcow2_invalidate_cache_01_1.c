static void qcow2_invalidate_cache(BlockDriverState *bs, Error **errp)

{

    BDRVQcowState *s = bs->opaque;

    int flags = s->flags;

    AES_KEY aes_encrypt_key;

    AES_KEY aes_decrypt_key;

    uint32_t crypt_method = 0;

    QDict *options;

    Error *local_err = NULL;

    int ret;



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



    bdrv_invalidate_cache(bs->file, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    memset(s, 0, sizeof(BDRVQcowState));

    options = qdict_clone_shallow(bs->options);



    ret = qcow2_open(bs, options, flags, &local_err);

    QDECREF(options);

    if (local_err) {

        error_setg(errp, "Could not reopen qcow2 layer: %s",

                   error_get_pretty(local_err));

        error_free(local_err);

        return;

    } else if (ret < 0) {

        error_setg_errno(errp, -ret, "Could not reopen qcow2 layer");

        return;

    }



    if (crypt_method) {

        s->crypt_method = crypt_method;

        memcpy(&s->aes_encrypt_key, &aes_encrypt_key, sizeof(aes_encrypt_key));

        memcpy(&s->aes_decrypt_key, &aes_decrypt_key, sizeof(aes_decrypt_key));

    }

}

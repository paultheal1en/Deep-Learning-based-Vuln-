qcrypto_block_luks_create(QCryptoBlock *block,

                          QCryptoBlockCreateOptions *options,

                          const char *optprefix,

                          QCryptoBlockInitFunc initfunc,

                          QCryptoBlockWriteFunc writefunc,

                          void *opaque,

                          Error **errp)

{

    QCryptoBlockLUKS *luks;

    QCryptoBlockCreateOptionsLUKS luks_opts;

    Error *local_err = NULL;

    uint8_t *masterkey = NULL;

    uint8_t *slotkey = NULL;

    uint8_t *splitkey = NULL;

    size_t splitkeylen = 0;

    size_t i;

    QCryptoCipher *cipher = NULL;

    QCryptoIVGen *ivgen = NULL;

    char *password;

    const char *cipher_alg;

    const char *cipher_mode;

    const char *ivgen_alg;

    const char *ivgen_hash_alg = NULL;

    const char *hash_alg;

    char *cipher_mode_spec = NULL;

    QCryptoCipherAlgorithm ivcipheralg = 0;

    uint64_t iters;



    memcpy(&luks_opts, &options->u.luks, sizeof(luks_opts));

    if (!luks_opts.has_iter_time) {

        luks_opts.iter_time = 2000;

    }

    if (!luks_opts.has_cipher_alg) {

        luks_opts.cipher_alg = QCRYPTO_CIPHER_ALG_AES_256;

    }

    if (!luks_opts.has_cipher_mode) {

        luks_opts.cipher_mode = QCRYPTO_CIPHER_MODE_XTS;

    }

    if (!luks_opts.has_ivgen_alg) {

        luks_opts.ivgen_alg = QCRYPTO_IVGEN_ALG_PLAIN64;

    }

    if (!luks_opts.has_hash_alg) {

        luks_opts.hash_alg = QCRYPTO_HASH_ALG_SHA256;

    }

    if (luks_opts.ivgen_alg == QCRYPTO_IVGEN_ALG_ESSIV) {

        if (!luks_opts.has_ivgen_hash_alg) {

            luks_opts.ivgen_hash_alg = QCRYPTO_HASH_ALG_SHA256;

            luks_opts.has_ivgen_hash_alg = true;

        }

    }

    /* Note we're allowing ivgen_hash_alg to be set even for

     * non-essiv iv generators that don't need a hash. It will

     * be silently ignored, for compatibility with dm-crypt */



    if (!options->u.luks.key_secret) {

        error_setg(errp, "Parameter '%skey-secret' is required for cipher",

                   optprefix ? optprefix : "");

        return -1;

    }

    password = qcrypto_secret_lookup_as_utf8(luks_opts.key_secret, errp);

    if (!password) {

        return -1;

    }



    luks = g_new0(QCryptoBlockLUKS, 1);

    block->opaque = luks;



    memcpy(luks->header.magic, qcrypto_block_luks_magic,

           QCRYPTO_BLOCK_LUKS_MAGIC_LEN);



    /* We populate the header in native endianness initially and

     * then convert everything to big endian just before writing

     * it out to disk

     */

    luks->header.version = QCRYPTO_BLOCK_LUKS_VERSION;

    qcrypto_block_luks_uuid_gen(luks->header.uuid);



    cipher_alg = qcrypto_block_luks_cipher_alg_lookup(luks_opts.cipher_alg,

                                                      errp);

    if (!cipher_alg) {

        goto error;

    }



    cipher_mode = QCryptoCipherMode_str(luks_opts.cipher_mode);

    ivgen_alg = QCryptoIVGenAlgorithm_str(luks_opts.ivgen_alg);

    if (luks_opts.has_ivgen_hash_alg) {

        ivgen_hash_alg = QCryptoHashAlgorithm_str(luks_opts.ivgen_hash_alg);

        cipher_mode_spec = g_strdup_printf("%s-%s:%s", cipher_mode, ivgen_alg,

                                           ivgen_hash_alg);

    } else {

        cipher_mode_spec = g_strdup_printf("%s-%s", cipher_mode, ivgen_alg);

    }

    hash_alg = QCryptoHashAlgorithm_str(luks_opts.hash_alg);





    if (strlen(cipher_alg) >= QCRYPTO_BLOCK_LUKS_CIPHER_NAME_LEN) {

        error_setg(errp, "Cipher name '%s' is too long for LUKS header",

                   cipher_alg);

        goto error;

    }

    if (strlen(cipher_mode_spec) >= QCRYPTO_BLOCK_LUKS_CIPHER_MODE_LEN) {

        error_setg(errp, "Cipher mode '%s' is too long for LUKS header",

                   cipher_mode_spec);

        goto error;

    }

    if (strlen(hash_alg) >= QCRYPTO_BLOCK_LUKS_HASH_SPEC_LEN) {

        error_setg(errp, "Hash name '%s' is too long for LUKS header",

                   hash_alg);

        goto error;

    }



    if (luks_opts.ivgen_alg == QCRYPTO_IVGEN_ALG_ESSIV) {

        ivcipheralg = qcrypto_block_luks_essiv_cipher(luks_opts.cipher_alg,

                                                      luks_opts.ivgen_hash_alg,

                                                      &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            goto error;

        }

    } else {

        ivcipheralg = luks_opts.cipher_alg;

    }



    strcpy(luks->header.cipher_name, cipher_alg);

    strcpy(luks->header.cipher_mode, cipher_mode_spec);

    strcpy(luks->header.hash_spec, hash_alg);



    luks->header.key_bytes = qcrypto_cipher_get_key_len(luks_opts.cipher_alg);

    if (luks_opts.cipher_mode == QCRYPTO_CIPHER_MODE_XTS) {

        luks->header.key_bytes *= 2;

    }



    /* Generate the salt used for hashing the master key

     * with PBKDF later

     */

    if (qcrypto_random_bytes(luks->header.master_key_salt,

                             QCRYPTO_BLOCK_LUKS_SALT_LEN,

                             errp) < 0) {

        goto error;

    }



    /* Generate random master key */

    masterkey = g_new0(uint8_t, luks->header.key_bytes);

    if (qcrypto_random_bytes(masterkey,

                             luks->header.key_bytes, errp) < 0) {

        goto error;

    }





    /* Setup the block device payload encryption objects */

    block->cipher = qcrypto_cipher_new(luks_opts.cipher_alg,

                                       luks_opts.cipher_mode,

                                       masterkey, luks->header.key_bytes,

                                       errp);

    if (!block->cipher) {

        goto error;

    }



    block->kdfhash = luks_opts.hash_alg;

    block->niv = qcrypto_cipher_get_iv_len(luks_opts.cipher_alg,

                                           luks_opts.cipher_mode);

    block->ivgen = qcrypto_ivgen_new(luks_opts.ivgen_alg,

                                     ivcipheralg,

                                     luks_opts.ivgen_hash_alg,

                                     masterkey, luks->header.key_bytes,

                                     errp);



    if (!block->ivgen) {

        goto error;

    }





    /* Determine how many iterations we need to hash the master

     * key, in order to have 1 second of compute time used

     */

    iters = qcrypto_pbkdf2_count_iters(luks_opts.hash_alg,

                                       masterkey, luks->header.key_bytes,

                                       luks->header.master_key_salt,

                                       QCRYPTO_BLOCK_LUKS_SALT_LEN,

                                       QCRYPTO_BLOCK_LUKS_DIGEST_LEN,

                                       &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto error;

    }



    if (iters > (ULLONG_MAX / luks_opts.iter_time)) {

        error_setg_errno(errp, ERANGE,

                         "PBKDF iterations %llu too large to scale",

                         (unsigned long long)iters);

        goto error;

    }



    /* iter_time was in millis, but count_iters reported for secs */

    iters = iters * luks_opts.iter_time / 1000;



    /* Why /= 8 ?  That matches cryptsetup, but there's no

     * explanation why they chose /= 8... Probably so that

     * if all 8 keyslots are active we only spend 1 second

     * in total time to check all keys */

    iters /= 8;

    if (iters > UINT32_MAX) {

        error_setg_errno(errp, ERANGE,

                         "PBKDF iterations %llu larger than %u",

                         (unsigned long long)iters, UINT32_MAX);

        goto error;

    }

    iters = MAX(iters, QCRYPTO_BLOCK_LUKS_MIN_MASTER_KEY_ITERS);

    luks->header.master_key_iterations = iters;



    /* Hash the master key, saving the result in the LUKS

     * header. This hash is used when opening the encrypted

     * device to verify that the user password unlocked a

     * valid master key

     */

    if (qcrypto_pbkdf2(luks_opts.hash_alg,

                       masterkey, luks->header.key_bytes,

                       luks->header.master_key_salt,

                       QCRYPTO_BLOCK_LUKS_SALT_LEN,

                       luks->header.master_key_iterations,

                       luks->header.master_key_digest,

                       QCRYPTO_BLOCK_LUKS_DIGEST_LEN,

                       errp) < 0) {

        goto error;

    }





    /* Although LUKS has multiple key slots, we're just going

     * to use the first key slot */

    splitkeylen = luks->header.key_bytes * QCRYPTO_BLOCK_LUKS_STRIPES;

    for (i = 0; i < QCRYPTO_BLOCK_LUKS_NUM_KEY_SLOTS; i++) {

        luks->header.key_slots[i].active = i == 0 ?

            QCRYPTO_BLOCK_LUKS_KEY_SLOT_ENABLED :

            QCRYPTO_BLOCK_LUKS_KEY_SLOT_DISABLED;

        luks->header.key_slots[i].stripes = QCRYPTO_BLOCK_LUKS_STRIPES;



        /* This calculation doesn't match that shown in the spec,

         * but instead follows the cryptsetup implementation.

         */

        luks->header.key_slots[i].key_offset =

            (QCRYPTO_BLOCK_LUKS_KEY_SLOT_OFFSET /

             QCRYPTO_BLOCK_LUKS_SECTOR_SIZE) +

            (ROUND_UP(DIV_ROUND_UP(splitkeylen, QCRYPTO_BLOCK_LUKS_SECTOR_SIZE),

                      (QCRYPTO_BLOCK_LUKS_KEY_SLOT_OFFSET /

                       QCRYPTO_BLOCK_LUKS_SECTOR_SIZE)) * i);

    }



    if (qcrypto_random_bytes(luks->header.key_slots[0].salt,

                             QCRYPTO_BLOCK_LUKS_SALT_LEN,

                             errp) < 0) {

        goto error;

    }



    /* Again we determine how many iterations are required to

     * hash the user password while consuming 1 second of compute

     * time */

    iters = qcrypto_pbkdf2_count_iters(luks_opts.hash_alg,

                                       (uint8_t *)password, strlen(password),

                                       luks->header.key_slots[0].salt,

                                       QCRYPTO_BLOCK_LUKS_SALT_LEN,

                                       luks->header.key_bytes,

                                       &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto error;

    }



    if (iters > (ULLONG_MAX / luks_opts.iter_time)) {

        error_setg_errno(errp, ERANGE,

                         "PBKDF iterations %llu too large to scale",

                         (unsigned long long)iters);

        goto error;

    }



    /* iter_time was in millis, but count_iters reported for secs */

    iters = iters * luks_opts.iter_time / 1000;



    if (iters > UINT32_MAX) {

        error_setg_errno(errp, ERANGE,

                         "PBKDF iterations %llu larger than %u",

                         (unsigned long long)iters, UINT32_MAX);

        goto error;

    }



    luks->header.key_slots[0].iterations =

        MAX(iters, QCRYPTO_BLOCK_LUKS_MIN_SLOT_KEY_ITERS);





    /* Generate a key that we'll use to encrypt the master

     * key, from the user's password

     */

    slotkey = g_new0(uint8_t, luks->header.key_bytes);

    if (qcrypto_pbkdf2(luks_opts.hash_alg,

                       (uint8_t *)password, strlen(password),

                       luks->header.key_slots[0].salt,

                       QCRYPTO_BLOCK_LUKS_SALT_LEN,

                       luks->header.key_slots[0].iterations,

                       slotkey, luks->header.key_bytes,

                       errp) < 0) {

        goto error;

    }





    /* Setup the encryption objects needed to encrypt the

     * master key material

     */

    cipher = qcrypto_cipher_new(luks_opts.cipher_alg,

                                luks_opts.cipher_mode,

                                slotkey, luks->header.key_bytes,

                                errp);

    if (!cipher) {

        goto error;

    }



    ivgen = qcrypto_ivgen_new(luks_opts.ivgen_alg,

                              ivcipheralg,

                              luks_opts.ivgen_hash_alg,

                              slotkey, luks->header.key_bytes,

                              errp);

    if (!ivgen) {

        goto error;

    }



    /* Before storing the master key, we need to vastly

     * increase its size, as protection against forensic

     * disk data recovery */

    splitkey = g_new0(uint8_t, splitkeylen);



    if (qcrypto_afsplit_encode(luks_opts.hash_alg,

                               luks->header.key_bytes,

                               luks->header.key_slots[0].stripes,

                               masterkey,

                               splitkey,

                               errp) < 0) {

        goto error;

    }



    /* Now we encrypt the split master key with the key generated

     * from the user's password, before storing it */

    if (qcrypto_block_encrypt_helper(cipher, block->niv, ivgen,

                                     QCRYPTO_BLOCK_LUKS_SECTOR_SIZE,

                                     0,

                                     splitkey,

                                     splitkeylen,

                                     errp) < 0) {

        goto error;

    }





    /* The total size of the LUKS headers is the partition header + key

     * slot headers, rounded up to the nearest sector, combined with

     * the size of each master key material region, also rounded up

     * to the nearest sector */

    luks->header.payload_offset =

        (QCRYPTO_BLOCK_LUKS_KEY_SLOT_OFFSET /

         QCRYPTO_BLOCK_LUKS_SECTOR_SIZE) +

        (ROUND_UP(DIV_ROUND_UP(splitkeylen, QCRYPTO_BLOCK_LUKS_SECTOR_SIZE),

                  (QCRYPTO_BLOCK_LUKS_KEY_SLOT_OFFSET /

                   QCRYPTO_BLOCK_LUKS_SECTOR_SIZE)) *

         QCRYPTO_BLOCK_LUKS_NUM_KEY_SLOTS);



    block->payload_offset = luks->header.payload_offset *

        QCRYPTO_BLOCK_LUKS_SECTOR_SIZE;



    /* Reserve header space to match payload offset */

    initfunc(block, block->payload_offset, opaque, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        goto error;

    }



    /* Everything on disk uses Big Endian, so flip header fields

     * before writing them */

    cpu_to_be16s(&luks->header.version);

    cpu_to_be32s(&luks->header.payload_offset);

    cpu_to_be32s(&luks->header.key_bytes);

    cpu_to_be32s(&luks->header.master_key_iterations);



    for (i = 0; i < QCRYPTO_BLOCK_LUKS_NUM_KEY_SLOTS; i++) {

        cpu_to_be32s(&luks->header.key_slots[i].active);

        cpu_to_be32s(&luks->header.key_slots[i].iterations);

        cpu_to_be32s(&luks->header.key_slots[i].key_offset);

        cpu_to_be32s(&luks->header.key_slots[i].stripes);

    }





    /* Write out the partition header and key slot headers */

    writefunc(block, 0,

              (const uint8_t *)&luks->header,

              sizeof(luks->header),

              opaque,

              &local_err);



    /* Delay checking local_err until we've byte-swapped */



    /* Byte swap the header back to native, in case we need

     * to read it again later */

    be16_to_cpus(&luks->header.version);

    be32_to_cpus(&luks->header.payload_offset);

    be32_to_cpus(&luks->header.key_bytes);

    be32_to_cpus(&luks->header.master_key_iterations);



    for (i = 0; i < QCRYPTO_BLOCK_LUKS_NUM_KEY_SLOTS; i++) {

        be32_to_cpus(&luks->header.key_slots[i].active);

        be32_to_cpus(&luks->header.key_slots[i].iterations);

        be32_to_cpus(&luks->header.key_slots[i].key_offset);

        be32_to_cpus(&luks->header.key_slots[i].stripes);

    }



    if (local_err) {

        error_propagate(errp, local_err);

        goto error;

    }



    /* Write out the master key material, starting at the

     * sector immediately following the partition header. */

    if (writefunc(block,

                  luks->header.key_slots[0].key_offset *

                  QCRYPTO_BLOCK_LUKS_SECTOR_SIZE,

                  splitkey, splitkeylen,

                  opaque,

                  errp) != splitkeylen) {

        goto error;

    }



    luks->cipher_alg = luks_opts.cipher_alg;

    luks->cipher_mode = luks_opts.cipher_mode;

    luks->ivgen_alg = luks_opts.ivgen_alg;

    luks->ivgen_hash_alg = luks_opts.ivgen_hash_alg;

    luks->hash_alg = luks_opts.hash_alg;



    memset(masterkey, 0, luks->header.key_bytes);

    g_free(masterkey);

    memset(slotkey, 0, luks->header.key_bytes);

    g_free(slotkey);

    g_free(splitkey);

    g_free(password);

    g_free(cipher_mode_spec);



    qcrypto_ivgen_free(ivgen);

    qcrypto_cipher_free(cipher);



    return 0;



 error:

    if (masterkey) {

        memset(masterkey, 0, luks->header.key_bytes);

    }

    g_free(masterkey);

    if (slotkey) {

        memset(slotkey, 0, luks->header.key_bytes);

    }

    g_free(slotkey);

    g_free(splitkey);

    g_free(password);

    g_free(cipher_mode_spec);



    qcrypto_ivgen_free(ivgen);

    qcrypto_cipher_free(cipher);



    g_free(luks);

    return -1;

}

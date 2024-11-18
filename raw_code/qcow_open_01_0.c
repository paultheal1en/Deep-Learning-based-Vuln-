static int qcow_open(BlockDriverState *bs, QDict *options, int flags,

                     Error **errp)

{

    BDRVQcowState *s = bs->opaque;

    unsigned int len, i, shift;

    int ret;

    QCowHeader header;

    Error *local_err = NULL;



    bs->file = bdrv_open_child(NULL, options, "file", bs, &child_file,

                               false, errp);

    if (!bs->file) {

        return -EINVAL;

    }



    ret = bdrv_pread(bs->file, 0, &header, sizeof(header));

    if (ret < 0) {

        goto fail;

    }

    be32_to_cpus(&header.magic);

    be32_to_cpus(&header.version);

    be64_to_cpus(&header.backing_file_offset);

    be32_to_cpus(&header.backing_file_size);

    be32_to_cpus(&header.mtime);

    be64_to_cpus(&header.size);

    be32_to_cpus(&header.crypt_method);

    be64_to_cpus(&header.l1_table_offset);



    if (header.magic != QCOW_MAGIC) {

        error_setg(errp, "Image not in qcow format");

        ret = -EINVAL;

        goto fail;

    }

    if (header.version != QCOW_VERSION) {

        error_setg(errp, "Unsupported qcow version %" PRIu32, header.version);

        ret = -ENOTSUP;

        goto fail;

    }



    if (header.size <= 1) {

        error_setg(errp, "Image size is too small (must be at least 2 bytes)");

        ret = -EINVAL;

        goto fail;

    }

    if (header.cluster_bits < 9 || header.cluster_bits > 16) {

        error_setg(errp, "Cluster size must be between 512 and 64k");

        ret = -EINVAL;

        goto fail;

    }



    /* l2_bits specifies number of entries; storing a uint64_t in each entry,

     * so bytes = num_entries << 3. */

    if (header.l2_bits < 9 - 3 || header.l2_bits > 16 - 3) {

        error_setg(errp, "L2 table size must be between 512 and 64k");

        ret = -EINVAL;

        goto fail;

    }



    if (header.crypt_method > QCOW_CRYPT_AES) {

        error_setg(errp, "invalid encryption method in qcow header");

        ret = -EINVAL;

        goto fail;

    }

    if (!qcrypto_cipher_supports(QCRYPTO_CIPHER_ALG_AES_128,

                                 QCRYPTO_CIPHER_MODE_CBC)) {

        error_setg(errp, "AES cipher not available");

        ret = -EINVAL;

        goto fail;

    }

    s->crypt_method_header = header.crypt_method;

    if (s->crypt_method_header) {

        if (bdrv_uses_whitelist() &&

            s->crypt_method_header == QCOW_CRYPT_AES) {

            error_setg(errp,

                       "Use of AES-CBC encrypted qcow images is no longer "

                       "supported in system emulators");

            error_append_hint(errp,

                              "You can use 'qemu-img convert' to convert your "

                              "image to an alternative supported format, such "

                              "as unencrypted qcow, or raw with the LUKS "

                              "format instead.\n");

            ret = -ENOSYS;

            goto fail;

        }



        bs->encrypted = true;

    }

    s->cluster_bits = header.cluster_bits;

    s->cluster_size = 1 << s->cluster_bits;

    s->cluster_sectors = 1 << (s->cluster_bits - 9);

    s->l2_bits = header.l2_bits;

    s->l2_size = 1 << s->l2_bits;

    bs->total_sectors = header.size / 512;

    s->cluster_offset_mask = (1LL << (63 - s->cluster_bits)) - 1;



    /* read the level 1 table */

    shift = s->cluster_bits + s->l2_bits;

    if (header.size > UINT64_MAX - (1LL << shift)) {

        error_setg(errp, "Image too large");

        ret = -EINVAL;

        goto fail;

    } else {

        uint64_t l1_size = (header.size + (1LL << shift) - 1) >> shift;

        if (l1_size > INT_MAX / sizeof(uint64_t)) {

            error_setg(errp, "Image too large");

            ret = -EINVAL;

            goto fail;

        }

        s->l1_size = l1_size;

    }



    s->l1_table_offset = header.l1_table_offset;

    s->l1_table = g_try_new(uint64_t, s->l1_size);

    if (s->l1_table == NULL) {

        error_setg(errp, "Could not allocate memory for L1 table");

        ret = -ENOMEM;

        goto fail;

    }



    ret = bdrv_pread(bs->file, s->l1_table_offset, s->l1_table,

               s->l1_size * sizeof(uint64_t));

    if (ret < 0) {

        goto fail;

    }



    for(i = 0;i < s->l1_size; i++) {

        be64_to_cpus(&s->l1_table[i]);

    }



    /* alloc L2 cache (max. 64k * 16 * 8 = 8 MB) */

    s->l2_cache =

        qemu_try_blockalign(bs->file->bs,

                            s->l2_size * L2_CACHE_SIZE * sizeof(uint64_t));

    if (s->l2_cache == NULL) {

        error_setg(errp, "Could not allocate L2 table cache");

        ret = -ENOMEM;

        goto fail;

    }

    s->cluster_cache = g_malloc(s->cluster_size);

    s->cluster_data = g_malloc(s->cluster_size);

    s->cluster_cache_offset = -1;



    /* read the backing file name */

    if (header.backing_file_offset != 0) {

        len = header.backing_file_size;

        if (len > 1023 || len >= sizeof(bs->backing_file)) {

            error_setg(errp, "Backing file name too long");

            ret = -EINVAL;

            goto fail;

        }

        ret = bdrv_pread(bs->file, header.backing_file_offset,

                   bs->backing_file, len);

        if (ret < 0) {

            goto fail;

        }

        bs->backing_file[len] = '\0';

    }



    /* Disable migration when qcow images are used */

    error_setg(&s->migration_blocker, "The qcow format used by node '%s' "

               "does not support live migration",

               bdrv_get_device_or_node_name(bs));

    ret = migrate_add_blocker(s->migration_blocker, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        error_free(s->migration_blocker);

        goto fail;

    }



    qemu_co_mutex_init(&s->lock);

    return 0;



 fail:

    g_free(s->l1_table);

    qemu_vfree(s->l2_cache);

    g_free(s->cluster_cache);

    g_free(s->cluster_data);

    return ret;

}

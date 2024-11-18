static int qcow2_do_open(BlockDriverState *bs, QDict *options, int flags,
                         Error **errp)
{
    BDRVQcow2State *s = bs->opaque;
    unsigned int len, i;
    int ret = 0;
    QCowHeader header;
    Error *local_err = NULL;
    uint64_t ext_end;
    uint64_t l1_vm_state_index;
    bool update_header = false;
    ret = bdrv_pread(bs->file, 0, &header, sizeof(header));
    if (ret < 0) {
        error_setg_errno(errp, -ret, "Could not read qcow2 header");
    be32_to_cpus(&header.magic);
    be32_to_cpus(&header.version);
    be64_to_cpus(&header.backing_file_offset);
    be32_to_cpus(&header.backing_file_size);
    be64_to_cpus(&header.size);
    be32_to_cpus(&header.cluster_bits);
    be32_to_cpus(&header.crypt_method);
    be64_to_cpus(&header.l1_table_offset);
    be32_to_cpus(&header.l1_size);
    be64_to_cpus(&header.refcount_table_offset);
    be32_to_cpus(&header.refcount_table_clusters);
    be64_to_cpus(&header.snapshots_offset);
    be32_to_cpus(&header.nb_snapshots);
    if (header.magic != QCOW_MAGIC) {
        error_setg(errp, "Image is not in qcow2 format");
    if (header.version < 2 || header.version > 3) {
        error_setg(errp, "Unsupported qcow2 version %" PRIu32, header.version);
        ret = -ENOTSUP;
    s->qcow_version = header.version;
    /* Initialise cluster size */
    if (header.cluster_bits < MIN_CLUSTER_BITS ||
        header.cluster_bits > MAX_CLUSTER_BITS) {
        error_setg(errp, "Unsupported cluster size: 2^%" PRIu32,
                   header.cluster_bits);
    s->cluster_bits = header.cluster_bits;
    s->cluster_size = 1 << s->cluster_bits;
    s->cluster_sectors = 1 << (s->cluster_bits - BDRV_SECTOR_BITS);
    /* Initialise version 3 header fields */
    if (header.version == 2) {
        header.incompatible_features    = 0;
        header.compatible_features      = 0;
        header.autoclear_features       = 0;
        header.refcount_order           = 4;
        header.header_length            = 72;
    } else {
        be64_to_cpus(&header.incompatible_features);
        be64_to_cpus(&header.compatible_features);
        be64_to_cpus(&header.autoclear_features);
        be32_to_cpus(&header.refcount_order);
        be32_to_cpus(&header.header_length);
        if (header.header_length < 104) {
            error_setg(errp, "qcow2 header too short");
    if (header.header_length > s->cluster_size) {
        error_setg(errp, "qcow2 header exceeds cluster size");
    if (header.header_length > sizeof(header)) {
        s->unknown_header_fields_size = header.header_length - sizeof(header);
        s->unknown_header_fields = g_malloc(s->unknown_header_fields_size);
        ret = bdrv_pread(bs->file, sizeof(header), s->unknown_header_fields,
                         s->unknown_header_fields_size);
        if (ret < 0) {
            error_setg_errno(errp, -ret, "Could not read unknown qcow2 header "
                             "fields");
    if (header.backing_file_offset > s->cluster_size) {
        error_setg(errp, "Invalid backing file offset");
    if (header.backing_file_offset) {
        ext_end = header.backing_file_offset;
    } else {
        ext_end = 1 << header.cluster_bits;
    /* Handle feature bits */
    s->incompatible_features    = header.incompatible_features;
    s->compatible_features      = header.compatible_features;
    s->autoclear_features       = header.autoclear_features;
    if (s->incompatible_features & ~QCOW2_INCOMPAT_MASK) {
        void *feature_table = NULL;
        qcow2_read_extensions(bs, header.header_length, ext_end,
                              &feature_table, flags, NULL, NULL);
        report_unsupported_feature(errp, feature_table,
                                   s->incompatible_features &
                                   ~QCOW2_INCOMPAT_MASK);
        ret = -ENOTSUP;
        g_free(feature_table);
    if (s->incompatible_features & QCOW2_INCOMPAT_CORRUPT) {
        /* Corrupt images may not be written to unless they are being repaired
         */
        if ((flags & BDRV_O_RDWR) && !(flags & BDRV_O_CHECK)) {
            error_setg(errp, "qcow2: Image is corrupt; cannot be opened "
                       "read/write");
            ret = -EACCES;
    /* Check support for various header values */
    if (header.refcount_order > 6) {
        error_setg(errp, "Reference count entry width too large; may not "
                   "exceed 64 bits");
    s->refcount_order = header.refcount_order;
    s->refcount_bits = 1 << s->refcount_order;
    s->refcount_max = UINT64_C(1) << (s->refcount_bits - 1);
    s->refcount_max += s->refcount_max - 1;
    s->crypt_method_header = header.crypt_method;
    if (s->crypt_method_header) {
        if (bdrv_uses_whitelist() &&
            s->crypt_method_header == QCOW_CRYPT_AES) {
            error_setg(errp,
                       "Use of AES-CBC encrypted qcow2 images is no longer "
                       "supported in system emulators");
            error_append_hint(errp,
                              "You can use 'qemu-img convert' to convert your "
                              "image to an alternative supported format, such "
                              "as unencrypted qcow2, or raw with the LUKS "
                              "format instead.\n");
            ret = -ENOSYS;
        if (s->crypt_method_header == QCOW_CRYPT_AES) {
            s->crypt_physical_offset = false;
        } else {
            /* Assuming LUKS and any future crypt methods we
             * add will all use physical offsets, due to the
             * fact that the alternative is insecure...  */
            s->crypt_physical_offset = true;
        bs->encrypted = true;
    s->l2_bits = s->cluster_bits - 3; /* L2 is always one cluster */
    s->l2_size = 1 << s->l2_bits;
    /* 2^(s->refcount_order - 3) is the refcount width in bytes */
    s->refcount_block_bits = s->cluster_bits - (s->refcount_order - 3);
    s->refcount_block_size = 1 << s->refcount_block_bits;
    bs->total_sectors = header.size / 512;
    s->csize_shift = (62 - (s->cluster_bits - 8));
    s->csize_mask = (1 << (s->cluster_bits - 8)) - 1;
    s->cluster_offset_mask = (1LL << s->csize_shift) - 1;
    s->refcount_table_offset = header.refcount_table_offset;
    s->refcount_table_size =
        header.refcount_table_clusters << (s->cluster_bits - 3);
    if (header.refcount_table_clusters > qcow2_max_refcount_clusters(s)) {
        error_setg(errp, "Reference count table too large");
    ret = validate_table_offset(bs, s->refcount_table_offset,
                                s->refcount_table_size, sizeof(uint64_t));
    if (ret < 0) {
        error_setg(errp, "Invalid reference count table offset");
    /* Snapshot table offset/length */
    if (header.nb_snapshots > QCOW_MAX_SNAPSHOTS) {
        error_setg(errp, "Too many snapshots");
    ret = validate_table_offset(bs, header.snapshots_offset,
                                header.nb_snapshots,
                                sizeof(QCowSnapshotHeader));
    if (ret < 0) {
        error_setg(errp, "Invalid snapshot table offset");
    /* read the level 1 table */
    if (header.l1_size > QCOW_MAX_L1_SIZE / sizeof(uint64_t)) {
        error_setg(errp, "Active L1 table too large");
        ret = -EFBIG;
    s->l1_size = header.l1_size;
    l1_vm_state_index = size_to_l1(s, header.size);
    if (l1_vm_state_index > INT_MAX) {
        error_setg(errp, "Image is too big");
        ret = -EFBIG;
    s->l1_vm_state_index = l1_vm_state_index;
    /* the L1 table must contain at least enough entries to put
       header.size bytes */
    if (s->l1_size < s->l1_vm_state_index) {
        error_setg(errp, "L1 table is too small");
    ret = validate_table_offset(bs, header.l1_table_offset,
                                header.l1_size, sizeof(uint64_t));
    if (ret < 0) {
        error_setg(errp, "Invalid L1 table offset");
    s->l1_table_offset = header.l1_table_offset;
    if (s->l1_size > 0) {
        s->l1_table = qemu_try_blockalign(bs->file->bs,
            align_offset(s->l1_size * sizeof(uint64_t), 512));
        if (s->l1_table == NULL) {
            error_setg(errp, "Could not allocate L1 table");
            ret = -ENOMEM;
        ret = bdrv_pread(bs->file, s->l1_table_offset, s->l1_table,
                         s->l1_size * sizeof(uint64_t));
        if (ret < 0) {
            error_setg_errno(errp, -ret, "Could not read L1 table");
        for(i = 0;i < s->l1_size; i++) {
            be64_to_cpus(&s->l1_table[i]);
    /* Parse driver-specific options */
    ret = qcow2_update_options(bs, options, flags, errp);
    if (ret < 0) {
    s->cluster_cache_offset = -1;
    s->flags = flags;
    ret = qcow2_refcount_init(bs);
    if (ret != 0) {
        error_setg_errno(errp, -ret, "Could not initialize refcount handling");
    QLIST_INIT(&s->cluster_allocs);
    QTAILQ_INIT(&s->discards);
    /* read qcow2 extensions */
    if (qcow2_read_extensions(bs, header.header_length, ext_end, NULL,
                              flags, &update_header, &local_err)) {
        error_propagate(errp, local_err);
    /* qcow2_read_extension may have set up the crypto context
     * if the crypt method needs a header region, some methods
     * don't need header extensions, so must check here
     */
    if (s->crypt_method_header && !s->crypto) {
        if (s->crypt_method_header == QCOW_CRYPT_AES) {
            unsigned int cflags = 0;
            if (flags & BDRV_O_NO_IO) {
                cflags |= QCRYPTO_BLOCK_OPEN_NO_IO;
            s->crypto = qcrypto_block_open(s->crypto_opts, "encrypt.",
                                           NULL, NULL, cflags, errp);
            if (!s->crypto) {
        } else if (!(flags & BDRV_O_NO_IO)) {
            error_setg(errp, "Missing CRYPTO header for crypt method %d",
                       s->crypt_method_header);
    /* read the backing file name */
    if (header.backing_file_offset != 0) {
        len = header.backing_file_size;
        if (len > MIN(1023, s->cluster_size - header.backing_file_offset) ||
            len >= sizeof(bs->backing_file)) {
            error_setg(errp, "Backing file name too long");
        ret = bdrv_pread(bs->file, header.backing_file_offset,
                         bs->backing_file, len);
        if (ret < 0) {
            error_setg_errno(errp, -ret, "Could not read backing file name");
        bs->backing_file[len] = '\0';
        s->image_backing_file = g_strdup(bs->backing_file);
    /* Internal snapshots */
    s->snapshots_offset = header.snapshots_offset;
    s->nb_snapshots = header.nb_snapshots;
    ret = qcow2_read_snapshots(bs);
    if (ret < 0) {
        error_setg_errno(errp, -ret, "Could not read snapshots");
    /* Clear unknown autoclear feature bits */
    update_header |= s->autoclear_features & ~QCOW2_AUTOCLEAR_MASK;
    update_header =
        update_header && !bs->read_only && !(flags & BDRV_O_INACTIVE);
    if (update_header) {
        s->autoclear_features &= QCOW2_AUTOCLEAR_MASK;
    if (qcow2_load_autoloading_dirty_bitmaps(bs, &local_err)) {
        update_header = false;
    if (local_err != NULL) {
        error_propagate(errp, local_err);
    if (update_header) {
        ret = qcow2_update_header(bs);
        if (ret < 0) {
            error_setg_errno(errp, -ret, "Could not update qcow2 header");
    /* Initialise locks */
    qemu_co_mutex_init(&s->lock);
    bs->supported_zero_flags = BDRV_REQ_MAY_UNMAP;
    /* Repair image if dirty */
    if (!(flags & (BDRV_O_CHECK | BDRV_O_INACTIVE)) && !bs->read_only &&
        (s->incompatible_features & QCOW2_INCOMPAT_DIRTY)) {
        BdrvCheckResult result = {0};
        ret = qcow2_check(bs, &result, BDRV_FIX_ERRORS | BDRV_FIX_LEAKS);
        if (ret < 0) {
            error_setg_errno(errp, -ret, "Could not repair dirty image");
#ifdef DEBUG_ALLOC
    {
        BdrvCheckResult result = {0};
        qcow2_check_refcounts(bs, &result, 0);
#endif
    return ret;
 fail:
    g_free(s->unknown_header_fields);
    cleanup_unknown_header_ext(bs);
    qcow2_free_snapshots(bs);
    qcow2_refcount_close(bs);
    qemu_vfree(s->l1_table);
    /* else pre-write overlap checks in cache_destroy may crash */
    s->l1_table = NULL;
    cache_clean_timer_del(bs);
    if (s->l2_table_cache) {
        qcow2_cache_destroy(bs, s->l2_table_cache);
    if (s->refcount_block_cache) {
        qcow2_cache_destroy(bs, s->refcount_block_cache);
    qcrypto_block_free(s->crypto);
    qapi_free_QCryptoBlockOpenOptions(s->crypto_opts);
    return ret;
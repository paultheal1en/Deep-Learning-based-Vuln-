static int bdrv_open_common(BlockDriverState *bs, BlockDriverState *file,
    QDict *options, int flags, BlockDriver *drv, Error **errp)
{
    int ret, open_flags;
    const char *filename;
    const char *node_name = NULL;
    Error *local_err = NULL;
    assert(drv != NULL);
    assert(bs->file == NULL);
    assert(options != NULL && bs->options != options);
    if (file != NULL) {
        filename = file->filename;
    } else {
        filename = qdict_get_try_str(options, "filename");
    if (drv->bdrv_needs_filename && !filename) {
        error_setg(errp, "The '%s' block driver requires a file name",
                   drv->format_name);
        return -EINVAL;
    trace_bdrv_open_common(bs, filename ?: "", flags, drv->format_name);
    node_name = qdict_get_try_str(options, "node-name");
    bdrv_assign_node_name(bs, node_name, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
        return -EINVAL;
    qdict_del(options, "node-name");
    /* bdrv_open() with directly using a protocol as drv. This layer is already
     * opened, so assign it to bs (while file becomes a closed BlockDriverState)
     * and return immediately. */
    if (file != NULL && drv->bdrv_file_open) {
        bdrv_swap(file, bs);
        return 0;
    bs->open_flags = flags;
    bs->guest_block_size = 512;
    bs->request_alignment = 512;
    bs->zero_beyond_eof = true;
    open_flags = bdrv_open_flags(bs, flags);
    bs->read_only = !(open_flags & BDRV_O_RDWR);
    if (use_bdrv_whitelist && !bdrv_is_whitelisted(drv, bs->read_only)) {
        error_setg(errp,
                   !bs->read_only && bdrv_is_whitelisted(drv, true)
                        ? "Driver '%s' can only be used for read-only devices"
                        : "Driver '%s' is not whitelisted",
                   drv->format_name);
        return -ENOTSUP;
    assert(bs->copy_on_read == 0); /* bdrv_new() and bdrv_close() make it so */
    if (flags & BDRV_O_COPY_ON_READ) {
        if (!bs->read_only) {
            bdrv_enable_copy_on_read(bs);
        } else {
            error_setg(errp, "Can't use copy-on-read on read-only device");
            return -EINVAL;
    if (filename != NULL) {
        pstrcpy(bs->filename, sizeof(bs->filename), filename);
    } else {
        bs->filename[0] = '\0';
    pstrcpy(bs->exact_filename, sizeof(bs->exact_filename), bs->filename);
    bs->drv = drv;
    bs->opaque = g_malloc0(drv->instance_size);
    bs->enable_write_cache = !!(flags & BDRV_O_CACHE_WB);
    /* Open the image, either directly or using a protocol */
    if (drv->bdrv_file_open) {
        assert(file == NULL);
        assert(!drv->bdrv_needs_filename || filename != NULL);
        ret = drv->bdrv_file_open(bs, options, open_flags, &local_err);
    } else {
        if (file == NULL) {
            error_setg(errp, "Can't use '%s' as a block driver for the "
                       "protocol level", drv->format_name);
            ret = -EINVAL;
            goto free_and_fail;
        bs->file = file;
        ret = drv->bdrv_open(bs, options, open_flags, &local_err);
    if (ret < 0) {
        if (local_err) {
            error_propagate(errp, local_err);
        } else if (bs->filename[0]) {
            error_setg_errno(errp, -ret, "Could not open '%s'", bs->filename);
        } else {
            error_setg_errno(errp, -ret, "Could not open image");
        goto free_and_fail;
    ret = refresh_total_sectors(bs, bs->total_sectors);
    if (ret < 0) {
        error_setg_errno(errp, -ret, "Could not refresh total sector count");
        goto free_and_fail;
    bdrv_refresh_limits(bs, &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
        ret = -EINVAL;
        goto free_and_fail;
    assert(bdrv_opt_mem_align(bs) != 0);
    assert((bs->request_alignment != 0) || bs->sg);
    return 0;
free_and_fail:
    bs->file = NULL;
    g_free(bs->opaque);
    bs->opaque = NULL;
    bs->drv = NULL;
    return ret;
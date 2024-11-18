static BlockBackend *blockdev_init(const char *file, QDict *bs_opts,
                                   Error **errp)
{
    const char *buf;
    int bdrv_flags = 0;
    int on_read_error, on_write_error;
    bool account_invalid, account_failed;
    BlockBackend *blk;
    BlockDriverState *bs;
    ThrottleConfig cfg;
    int snapshot = 0;
    Error *error = NULL;
    QemuOpts *opts;
    const char *id;
    bool has_driver_specific_opts;
    BlockdevDetectZeroesOptions detect_zeroes =
        BLOCKDEV_DETECT_ZEROES_OPTIONS_OFF;
    const char *throttling_group = NULL;
    /* Check common options by copying from bs_opts to opts, all other options
     * stay in bs_opts for processing by bdrv_open(). */
    id = qdict_get_try_str(bs_opts, "id");
    opts = qemu_opts_create(&qemu_common_drive_opts, id, 1, &error);
    if (error) {
        error_propagate(errp, error);
        goto err_no_opts;
    }
    qemu_opts_absorb_qdict(opts, bs_opts, &error);
    if (error) {
        error_propagate(errp, error);
        goto early_err;
    }
    if (id) {
        qdict_del(bs_opts, "id");
    }
    has_driver_specific_opts = !!qdict_size(bs_opts);
    /* extract parameters */
    snapshot = qemu_opt_get_bool(opts, "snapshot", 0);
    extract_common_blockdev_options(opts, &bdrv_flags, &throttling_group, &cfg,
                                    &detect_zeroes, &error);
    if (error) {
        error_propagate(errp, error);
        goto early_err;
    }
    if ((buf = qemu_opt_get(opts, "format")) != NULL) {
        if (is_help_option(buf)) {
            error_printf("Supported formats:");
            bdrv_iterate_format(bdrv_format_print, NULL);
            error_printf("\n");
            goto early_err;
        }
        if (qdict_haskey(bs_opts, "driver")) {
            error_setg(errp, "Cannot specify both 'driver' and 'format'");
            goto early_err;
        }
        qdict_put(bs_opts, "driver", qstring_from_str(buf));
    }
    on_write_error = BLOCKDEV_ON_ERROR_ENOSPC;
    if ((buf = qemu_opt_get(opts, "werror")) != NULL) {
        on_write_error = parse_block_error_action(buf, 0, &error);
        if (error) {
            error_propagate(errp, error);
            goto early_err;
        }
    }
    on_read_error = BLOCKDEV_ON_ERROR_REPORT;
    if ((buf = qemu_opt_get(opts, "rerror")) != NULL) {
        on_read_error = parse_block_error_action(buf, 1, &error);
        if (error) {
            error_propagate(errp, error);
            goto early_err;
        }
    }
    if (snapshot) {
        /* always use cache=unsafe with snapshot */
        bdrv_flags &= ~BDRV_O_CACHE_MASK;
        bdrv_flags |= (BDRV_O_SNAPSHOT|BDRV_O_CACHE_WB|BDRV_O_NO_FLUSH);
    }
    /* init */
    if ((!file || !*file) && !has_driver_specific_opts) {
        BlockBackendRootState *blk_rs;
        blk = blk_new(qemu_opts_id(opts), errp);
        if (!blk) {
            goto early_err;
        }
        blk_rs = blk_get_root_state(blk);
        blk_rs->open_flags    = bdrv_flags;
        blk_rs->read_only     = !(bdrv_flags & BDRV_O_RDWR);
        blk_rs->detect_zeroes = detect_zeroes;
        if (throttle_enabled(&cfg)) {
            if (!throttling_group) {
                throttling_group = blk_name(blk);
            }
            blk_rs->throttle_group = g_strdup(throttling_group);
            blk_rs->throttle_state = throttle_group_incref(throttling_group);
            blk_rs->throttle_state->cfg = cfg;
        }
        QDECREF(bs_opts);
    } else {
        if (file && !*file) {
            file = NULL;
        }
        blk = blk_new_open(qemu_opts_id(opts), file, NULL, bs_opts, bdrv_flags,
                           errp);
        if (!blk) {
            goto err_no_bs_opts;
        }
        bs = blk_bs(blk);
        bs->detect_zeroes = detect_zeroes;
        /* disk I/O throttling */
        if (throttle_enabled(&cfg)) {
            if (!throttling_group) {
                throttling_group = blk_name(blk);
            }
            bdrv_io_limits_enable(bs, throttling_group);
            bdrv_set_io_limits(bs, &cfg);
        }
        if (bdrv_key_required(bs)) {
            autostart = 0;
        }
        block_acct_init(blk_get_stats(blk), account_invalid, account_failed);
    }
    blk_set_on_error(blk, on_read_error, on_write_error);
err_no_bs_opts:
    qemu_opts_del(opts);
    return blk;
early_err:
    qemu_opts_del(opts);
err_no_opts:
    QDECREF(bs_opts);
    return NULL;
}
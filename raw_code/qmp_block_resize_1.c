void qmp_block_resize(bool has_device, const char *device,
                      bool has_node_name, const char *node_name,
                      int64_t size, Error **errp)
{
    Error *local_err = NULL;
    BlockDriverState *bs;
    int ret;
    bs = bdrv_lookup_bs(has_device ? device : NULL,
                        has_node_name ? node_name : NULL,
                        &local_err);
    if (local_err) {
        error_propagate(errp, local_err);
    if (!bdrv_is_first_non_filter(bs)) {
        error_set(errp, QERR_FEATURE_DISABLED, "resize");
    if (size < 0) {
        error_set(errp, QERR_INVALID_PARAMETER_VALUE, "size", "a >0 size");
    /* complete all in-flight operations before resizing the device */
    bdrv_drain_all();
    ret = bdrv_truncate(bs, size);
    switch (ret) {
    case 0:
        break;
    case -ENOMEDIUM:
        error_set(errp, QERR_DEVICE_HAS_NO_MEDIUM, device);
        break;
    case -ENOTSUP:
        error_set(errp, QERR_UNSUPPORTED);
        break;
    case -EACCES:
        error_set(errp, QERR_DEVICE_IS_READ_ONLY, device);
        break;
    case -EBUSY:
        break;
    default:
        error_setg_errno(errp, -ret, "Could not resize");
        break;
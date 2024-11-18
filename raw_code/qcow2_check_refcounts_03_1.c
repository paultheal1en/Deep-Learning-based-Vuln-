int qcow2_check_refcounts(BlockDriverState *bs, BdrvCheckResult *res,
                          BdrvCheckMode fix)
{
    BDRVQcowState *s = bs->opaque;
    BdrvCheckResult pre_compare_res;
    int64_t size, highest_cluster, nb_clusters;
    uint16_t *refcount_table = NULL;
    bool rebuild = false;
    int ret;
    size = bdrv_getlength(bs->file);
    if (size < 0) {
        res->check_errors++;
        return size;
    nb_clusters = size_to_clusters(s, size);
    if (nb_clusters > INT_MAX) {
        res->check_errors++;
        return -EFBIG;
    res->bfi.total_clusters =
        size_to_clusters(s, bs->total_sectors * BDRV_SECTOR_SIZE);
    ret = calculate_refcounts(bs, res, fix, &rebuild, &refcount_table,
    /* In case we don't need to rebuild the refcount structure (but want to fix
     * something), this function is immediately called again, in which case the
     * result should be ignored */
    pre_compare_res = *res;
    compare_refcounts(bs, res, 0, &rebuild, &highest_cluster, refcount_table,
                      nb_clusters);
    if (rebuild && (fix & BDRV_FIX_ERRORS)) {
        fprintf(stderr, "Rebuilding refcount structure\n");
        ret = rebuild_refcount_structure(bs, res, &refcount_table,
    } else if (fix) {
            fprintf(stderr, "ERROR need to rebuild refcount structures\n");
            res->check_errors++;
            ret = -EIO;
        if (res->leaks || res->corruptions) {
            *res = pre_compare_res;
            compare_refcounts(bs, res, fix, &rebuild, &highest_cluster,
                              refcount_table, nb_clusters);
    /* check OFLAG_COPIED */
    ret = check_oflag_copied(bs, res, fix);
    res->image_end_offset = (highest_cluster + 1) * s->cluster_size;
    ret = 0;
fail:
    g_free(refcount_table);
    return ret;
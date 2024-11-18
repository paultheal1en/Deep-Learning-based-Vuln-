int64_t xbzrle_cache_resize(int64_t new_size, Error **errp)

{

    PageCache *new_cache;

    int64_t ret;



    /* Check for truncation */

    if (new_size != (size_t)new_size) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "cache size",

                   "exceeding address space");

        return -1;

    }



    /* Cache should not be larger than guest ram size */

    if (new_size > ram_bytes_total()) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "cache size",

                   "exceeds guest ram size");

        return -1;

    }



    XBZRLE_cache_lock();



    if (XBZRLE.cache != NULL) {

        if (pow2floor(new_size) == migrate_xbzrle_cache_size()) {

            goto out_new_size;

        }

        new_cache = cache_init(new_size, TARGET_PAGE_SIZE, errp);

        if (!new_cache) {

            ret = -1;

            goto out;

        }



        cache_fini(XBZRLE.cache);

        XBZRLE.cache = new_cache;

    }



out_new_size:

    ret = pow2floor(new_size);

out:

    XBZRLE_cache_unlock();

    return ret;

}

int64_t xbzrle_cache_resize(int64_t new_size)

{

    if (new_size < TARGET_PAGE_SIZE) {

        return -1;

    }



    if (XBZRLE.cache != NULL) {

        return cache_resize(XBZRLE.cache, new_size / TARGET_PAGE_SIZE) *

            TARGET_PAGE_SIZE;

    }

    return pow2floor(new_size);

}

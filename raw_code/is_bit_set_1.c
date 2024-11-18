static inline int is_bit_set(BlockDriverState *bs, int64_t bitnum)

{

    uint64_t offset = sizeof(struct cow_header_v2) + bitnum / 8;

    uint8_t bitmap;



    if (bdrv_pread(bs->file, offset, &bitmap, sizeof(bitmap)) !=

	    sizeof(bitmap)) {

       return -errno;

    }



    return !!(bitmap & (1 << (bitnum % 8)));

}

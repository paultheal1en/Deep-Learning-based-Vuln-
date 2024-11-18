static int bdrv_check_byte_request(BlockDriverState *bs, int64_t offset,

                                   size_t size)

{

    int64_t len;



    if (size > INT_MAX) {

        return -EIO;

    }



    if (!bdrv_is_inserted(bs))

        return -ENOMEDIUM;



    if (bs->growable)

        return 0;



    len = bdrv_getlength(bs);



    if (offset < 0)

        return -EIO;



    if ((offset > len) || (len - offset < size))

        return -EIO;



    return 0;

}

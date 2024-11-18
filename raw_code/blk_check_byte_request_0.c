static int blk_check_byte_request(BlockBackend *blk, int64_t offset,

                                  size_t size)

{

    int64_t len;



    if (size > INT_MAX) {

        return -EIO;

    }



    if (!blk_is_available(blk)) {

        return -ENOMEDIUM;

    }



    len = blk_getlength(blk);

    if (len < 0) {

        return len;

    }



    if (offset < 0) {

        return -EIO;

    }



    if (offset > len || len - offset < size) {

        return -EIO;

    }



    return 0;

}

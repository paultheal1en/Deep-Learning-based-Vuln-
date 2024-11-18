static int create_fixed_disk(int fd, uint8_t *buf, int64_t total_size)

{

    int ret = -EIO;



    /* Add footer to total size */

    total_size += 512;

    if (ftruncate(fd, total_size) != 0) {

        ret = -errno;

        goto fail;

    }

    if (lseek(fd, -512, SEEK_END) < 0) {

        goto fail;

    }

    if (write(fd, buf, HEADER_SIZE) != HEADER_SIZE) {

        goto fail;

    }



    ret = 0;



 fail:

    return ret;

}

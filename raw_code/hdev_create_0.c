static int hdev_create(const char *filename, QEMUOptionParameter *options)

{

    int fd;

    int ret = 0;

    struct stat stat_buf;

    int64_t total_size = 0;



    /* Read out options */

    while (options && options->name) {

        if (!strcmp(options->name, "size")) {

            total_size = options->value.n / 512;

        }

        options++;

    }



    fd = open(filename, O_WRONLY | O_BINARY);

    if (fd < 0)

        return -EIO;



    if (fstat(fd, &stat_buf) < 0)

        ret = -EIO;

    else if (!S_ISBLK(stat_buf.st_mode))

        ret = -EIO;

    else if (lseek(fd, 0, SEEK_END) < total_size * 512)

        ret = -ENOSPC;



    close(fd);

    return ret;

}

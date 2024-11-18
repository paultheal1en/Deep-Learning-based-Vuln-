static int ffm_write_write_index(int fd, int64_t pos)

{

    uint8_t buf[8];

    int i;



    for(i=0;i<8;i++)

        buf[i] = (pos >> (56 - i * 8)) & 0xff;

    lseek(fd, 8, SEEK_SET);

    if (write(fd, buf, 8) != 8)

        return AVERROR(EIO);

    return 8;

}

static int read_object(int fd, char *buf, uint64_t oid, int copies,

                       unsigned int datalen, uint64_t offset, bool cache)

{

    return read_write_object(fd, buf, oid, copies, datalen, offset, false,

                             false, cache);

}

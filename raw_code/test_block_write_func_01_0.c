static ssize_t test_block_write_func(QCryptoBlock *block,

                                     void *opaque,

                                     size_t offset,

                                     const uint8_t *buf,

                                     size_t buflen,

                                     Error **errp)

{

    Buffer *header = opaque;



    g_assert_cmpint(buflen + offset, <=, header->capacity);



    memcpy(header->buffer + offset, buf, buflen);

    header->offset = offset + buflen;



    return buflen;

}

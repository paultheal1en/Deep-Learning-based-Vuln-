static int cow_probe(const uint8_t *buf, int buf_size, const char *filename)

{

    const struct cow_header_v2 *cow_header = (const void *)buf;



    if (buf_size >= sizeof(struct cow_header_v2) &&

        be32_to_cpu(cow_header->magic) == COW_MAGIC &&

        be32_to_cpu(cow_header->version) == COW_VERSION)

        return 100;

    else

        return 0;

}

static int migration_put_buffer(void *opaque, const uint8_t *buf,

                               int64_t pos, int size)

{

    MigrationState *s = opaque;

    int ret;



    DPRINTF("putting %d bytes at %" PRId64 "\n", size, pos);



    if (size <= 0) {

        return size;

    }



    qemu_put_buffer(s->migration_file, buf, size);

    ret = qemu_file_get_error(s->migration_file);

    if (ret) {

        return ret;

    }



    s->bytes_xfer += size;

    return size;

}

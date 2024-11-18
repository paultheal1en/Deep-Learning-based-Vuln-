static void bdrv_put_buffer(void *opaque, const uint8_t *buf,

                            int64_t pos, int size)

{

    QEMUFileBdrv *s = opaque;

    bdrv_pwrite(s->bs, s->base_offset + pos, buf, size);

}

static void put_unused_buffer(QEMUFile *f, void *pv, size_t size)

{

    static const uint8_t buf[1024];

    int block_len;



    while (size > 0) {

        block_len = MIN(sizeof(buf), size);

        size -= block_len;

        qemu_put_buffer(f, buf, block_len);

    }

}

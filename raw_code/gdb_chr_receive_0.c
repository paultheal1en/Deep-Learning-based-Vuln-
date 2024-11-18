static void gdb_chr_receive(void *opaque, const uint8_t *buf, int size)

{

    GDBState *s = opaque;

    int i;



    for (i = 0; i < size; i++) {

        gdb_read_byte(s, buf[i]);

    }

}

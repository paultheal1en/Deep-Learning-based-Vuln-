static void buffer_release(void *opaque, uint8_t *data)

{

    *(uint8_t*)opaque = 0;

}

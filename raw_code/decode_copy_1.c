static int decode_copy(uint8_t *frame, int width, int height,

                       const uint8_t *src, const uint8_t *src_end)

{

    const int size = width * height;



    if (src_end - src < size)

        return -1;

    bytestream_get_buffer(&src, frame, size);

    return 0;

}

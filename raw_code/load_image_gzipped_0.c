int load_image_gzipped(const char *filename, hwaddr addr, uint64_t max_sz)

{

    uint8_t *compressed_data = NULL;

    uint8_t *data = NULL;

    gsize len;

    ssize_t bytes;

    int ret = -1;



    if (!g_file_get_contents(filename, (char **) &compressed_data, &len,

                             NULL)) {

        goto out;

    }



    /* Is it a gzip-compressed file? */

    if (len < 2 ||

        compressed_data[0] != 0x1f ||

        compressed_data[1] != 0x8b) {

        goto out;

    }



    if (max_sz > LOAD_IMAGE_MAX_GUNZIP_BYTES) {

        max_sz = LOAD_IMAGE_MAX_GUNZIP_BYTES;

    }



    data = g_malloc(max_sz);

    bytes = gunzip(data, max_sz, compressed_data, len);

    if (bytes < 0) {

        fprintf(stderr, "%s: unable to decompress gzipped kernel file\n",

                filename);

        goto out;

    }



    rom_add_blob_fixed(filename, data, bytes, addr);

    ret = bytes;



 out:

    g_free(compressed_data);

    g_free(data);

    return ret;

}

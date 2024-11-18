static int read_rle_sgi(const SGIInfo *sgi_info, 

        AVPicture *pict, ByteIOContext *f)

{

    uint8_t *dest_row, *rle_data = NULL;

    unsigned long *start_table, *length_table;

    int y, z, xsize, ysize, zsize, tablen; 

    long start_offset, run_length;

    int ret = 0;



    xsize = sgi_info->xsize;

    ysize = sgi_info->ysize;

    zsize = sgi_info->zsize;



    rle_data = av_malloc(xsize);



    /* skip header */ 

    url_fseek(f, SGI_HEADER_SIZE, SEEK_SET);



    /* size of rle offset and length tables */

    tablen = ysize * zsize * sizeof(long);



    start_table = (unsigned long *)av_malloc(tablen);

    length_table = (unsigned long *)av_malloc(tablen);



    if (!get_buffer(f, (uint8_t *)start_table, tablen)) {

        ret = -1;

        goto fail;

    }



    if (!get_buffer(f, (uint8_t *)length_table, tablen)) {

        ret = -1;

        goto fail;

    }



    for (z = 0; z < zsize; z++) {

        for (y = 0; y < ysize; y++) {

            dest_row = pict->data[0] + (ysize - 1 - y) * (xsize * zsize);



            start_offset = BE_32(&start_table[y + z * ysize]);

            run_length = BE_32(&length_table[y + z * ysize]);



            /* don't seek if already in the correct spot */

            if (url_ftell(f) != start_offset) {

                url_fseek(f, start_offset, SEEK_SET);

            }



            get_buffer(f, rle_data, run_length);

            

            expand_rle_row(dest_row, rle_data, z, zsize);

        }

    }



fail:

    av_free(start_table);

    av_free(length_table);

    av_free(rle_data);



    return ret;

}

static inline void libopenjpeg_copy_to_packed16(AVFrame *picture, opj_image_t *image) {

    uint16_t *img_ptr;

    int index, x, y, c;

    int adjust[4];

    for (x = 0; x < image->numcomps; x++)

        adjust[x] = FFMAX(FFMIN(av_pix_fmt_desc_get(picture->format)->comp[x].depth_minus1 + 1 - image->comps[x].prec, 8), 0);



    for (y = 0; y < picture->height; y++) {

        index = y*picture->width;

        img_ptr = (uint16_t*) (picture->data[0] + y*picture->linesize[0]);

        for (x = 0; x < picture->width; x++, index++) {

            for (c = 0; c < image->numcomps; c++) {

                *img_ptr++ = 0x8000 * image->comps[c].sgnd + (image->comps[c].data[index] << adjust[c]);

            }

        }

    }

}

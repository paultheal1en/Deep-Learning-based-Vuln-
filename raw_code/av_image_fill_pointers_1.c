int av_image_fill_pointers(uint8_t *data[4], enum PixelFormat pix_fmt, int height,

                           uint8_t *ptr, const int linesizes[4])

{

    int i, total_size, size[4], has_plane[4];



    const AVPixFmtDescriptor *desc = &av_pix_fmt_descriptors[pix_fmt];

    memset(data     , 0, sizeof(data[0])*4);

    memset(size     , 0, sizeof(size));

    memset(has_plane, 0, sizeof(has_plane));



    if ((unsigned)pix_fmt >= PIX_FMT_NB || desc->flags & PIX_FMT_HWACCEL)

        return AVERROR(EINVAL);



    data[0] = ptr;

    if (linesizes[0] > (INT_MAX - 1024) / height)

        return AVERROR(EINVAL);

    size[0] = linesizes[0] * height;



    if (desc->flags & PIX_FMT_PAL) {

        size[0] = (size[0] + 3) & ~3;

        data[1] = ptr + size[0]; /* palette is stored here as 256 32 bits words */

        return size[0] + 256 * 4;

    }



    for (i = 0; i < 4; i++)

        has_plane[desc->comp[i].plane] = 1;



    total_size = size[0];

    for (i = 1; i < 4 && has_plane[i]; i++) {

        int h, s = (i == 1 || i == 2) ? desc->log2_chroma_h : 0;

        data[i] = data[i-1] + size[i-1];

        h = (height + (1 << s) - 1) >> s;

        if (linesizes[i] > INT_MAX / h)

            return AVERROR(EINVAL);

        size[i] = h * linesizes[i];

        if (total_size > INT_MAX - size[i])

            return AVERROR(EINVAL);

        total_size += size[i];

    }



    return total_size;

}

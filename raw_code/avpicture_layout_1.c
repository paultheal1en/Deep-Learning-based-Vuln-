int avpicture_layout(const AVPicture* src, int pix_fmt, int width, int height,

                     unsigned char *dest, int dest_size)

{

    PixFmtInfo* pf = &pix_fmt_info[pix_fmt];

    int i, j, w, h, data_planes;

    const unsigned char* s; 

    int size = avpicture_get_size(pix_fmt, width, height);



    if (size > dest_size)

        return -1;



    if (pf->pixel_type == FF_PIXEL_PACKED || pf->pixel_type == FF_PIXEL_PALETTE) {

        if (pix_fmt == PIX_FMT_YUV422 || 

            pix_fmt == PIX_FMT_UYVY422 || 

            pix_fmt == PIX_FMT_RGB565 ||

            pix_fmt == PIX_FMT_RGB555)

            w = width * 2;

	else if (pix_fmt == PIX_FMT_UYVY411)

	  w = width + width/2;

	else if (pix_fmt == PIX_FMT_PAL8)

	  w = width;

	else

	  w = width * (pf->depth * pf->nb_channels / 8);

	  

	data_planes = 1;

	h = height;

    } else {

        data_planes = pf->nb_channels;

	w = (width*pf->depth + 7)/8;

	h = height;

    }

    

    for (i=0; i<data_planes; i++) {

         if (i == 1) {

	     w = width >> pf->x_chroma_shift;

	     h = height >> pf->y_chroma_shift;

	 }

         s = src->data[i];

	 for(j=0; j<h; j++) {

	     memcpy(dest, s, w);

	     dest += w;

	     s += src->linesize[i];

	 }

    }

    

    if (pf->pixel_type == FF_PIXEL_PALETTE)

	memcpy((unsigned char *)(((size_t)dest + 3) & ~3), src->data[1], 256 * 4);

    

    return size;

}

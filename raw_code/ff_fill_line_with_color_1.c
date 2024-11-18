int ff_fill_line_with_color(uint8_t *line[4], int pixel_step[4], int w, uint8_t dst_color[4],

                            enum AVPixelFormat pix_fmt, uint8_t rgba_color[4],

                            int *is_packed_rgba, uint8_t rgba_map_ptr[4])

{

    uint8_t rgba_map[4] = {0};

    int i;

    const AVPixFmtDescriptor *pix_desc = av_pix_fmt_desc_get(pix_fmt);

    int hsub = pix_desc->log2_chroma_w;



    *is_packed_rgba = ff_fill_rgba_map(rgba_map, pix_fmt) >= 0;



    if (*is_packed_rgba) {

        pixel_step[0] = (av_get_bits_per_pixel(pix_desc))>>3;

        for (i = 0; i < 4; i++)

            dst_color[rgba_map[i]] = rgba_color[i];



        line[0] = av_malloc_array(w, pixel_step[0]);



        for (i = 0; i < w; i++)

            memcpy(line[0] + i * pixel_step[0], dst_color, pixel_step[0]);

        if (rgba_map_ptr)

            memcpy(rgba_map_ptr, rgba_map, sizeof(rgba_map[0]) * 4);

    } else {

        int plane;



        dst_color[0] = RGB_TO_Y_CCIR(rgba_color[0], rgba_color[1], rgba_color[2]);

        dst_color[1] = RGB_TO_U_CCIR(rgba_color[0], rgba_color[1], rgba_color[2], 0);

        dst_color[2] = RGB_TO_V_CCIR(rgba_color[0], rgba_color[1], rgba_color[2], 0);

        dst_color[3] = rgba_color[3];



        for (plane = 0; plane < 4; plane++) {

            int line_size;

            int hsub1 = (plane == 1 || plane == 2) ? hsub : 0;



            pixel_step[plane] = 1;

            line_size = FF_CEIL_RSHIFT(w, hsub1) * pixel_step[plane];

            line[plane] = av_malloc(line_size);

            if (!line[plane]) {

                while(plane && line[plane-1])

                    av_freep(&line[--plane]);


            }

            memset(line[plane], dst_color[plane], line_size);

        }

    }



    return 0;

}
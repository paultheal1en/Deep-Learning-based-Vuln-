int ff_get_schro_frame_format (SchroChromaFormat schro_pix_fmt,

                               SchroFrameFormat  *schro_frame_fmt)

{

    unsigned int num_formats = sizeof(schro_pixel_format_map) /

                               sizeof(schro_pixel_format_map[0]);



    int idx;



    for (idx = 0; idx < num_formats; ++idx) {

        if (schro_pixel_format_map[idx].schro_pix_fmt == schro_pix_fmt) {

            *schro_frame_fmt = schro_pixel_format_map[idx].schro_frame_fmt;

            return 0;

        }

    }

    return -1;

}

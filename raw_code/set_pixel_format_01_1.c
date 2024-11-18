static void set_pixel_format(VncState *vs,

			     int bits_per_pixel, int depth,

			     int big_endian_flag, int true_color_flag,

			     int red_max, int green_max, int blue_max,

			     int red_shift, int green_shift, int blue_shift)

{

    int host_big_endian_flag;



#ifdef WORDS_BIGENDIAN

    host_big_endian_flag = 1;

#else

    host_big_endian_flag = 0;

#endif

    if (!true_color_flag) {

    fail:

	vnc_client_error(vs);

        return;

    }

    if (bits_per_pixel == 32 &&

        bits_per_pixel == vs->depth * 8 &&

        host_big_endian_flag == big_endian_flag &&

        red_max == 0xff && green_max == 0xff && blue_max == 0xff &&

        red_shift == 16 && green_shift == 8 && blue_shift == 0) {

        vs->depth = 4;

        vs->write_pixels = vnc_write_pixels_copy;

        vs->send_hextile_tile = send_hextile_tile_32;

    } else

    if (bits_per_pixel == 16 &&

        bits_per_pixel == vs->depth * 8 && 

        host_big_endian_flag == big_endian_flag &&

        red_max == 31 && green_max == 63 && blue_max == 31 &&

        red_shift == 11 && green_shift == 5 && blue_shift == 0) {

        vs->depth = 2;

        vs->write_pixels = vnc_write_pixels_copy;

        vs->send_hextile_tile = send_hextile_tile_16;

    } else

    if (bits_per_pixel == 8 &&

        bits_per_pixel == vs->depth * 8 &&

        red_max == 7 && green_max == 7 && blue_max == 3 &&

        red_shift == 5 && green_shift == 2 && blue_shift == 0) {

        vs->depth = 1;

        vs->write_pixels = vnc_write_pixels_copy;

        vs->send_hextile_tile = send_hextile_tile_8;

    } else

    {

        /* generic and slower case */

        if (bits_per_pixel != 8 &&

            bits_per_pixel != 16 &&

            bits_per_pixel != 32)

            goto fail;

        if (vs->depth == 4) {

            vs->send_hextile_tile = send_hextile_tile_generic_32;

        } else if (vs->depth == 2) {

           vs->send_hextile_tile = send_hextile_tile_generic_16;

        } else {

            vs->send_hextile_tile = send_hextile_tile_generic_8;

        }



        vs->pix_big_endian = big_endian_flag;

        vs->write_pixels = vnc_write_pixels_generic;

    }



    vs->client_red_shift = red_shift;

    vs->client_red_max = red_max;

    vs->client_green_shift = green_shift;

    vs->client_green_max = green_max;

    vs->client_blue_shift = blue_shift;

    vs->client_blue_max = blue_max;

    vs->pix_bpp = bits_per_pixel / 8;



    vga_hw_invalidate();

    vga_hw_update();

}

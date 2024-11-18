static void pixel_format_message (VncState *vs) {

    char pad[3] = { 0, 0, 0 };



    vnc_write_u8(vs, vs->depth * 8); /* bits-per-pixel */

    if (vs->depth == 4) vnc_write_u8(vs, 24); /* depth */

    else vnc_write_u8(vs, vs->depth * 8); /* depth */



#ifdef WORDS_BIGENDIAN

    vnc_write_u8(vs, 1);             /* big-endian-flag */

#else

    vnc_write_u8(vs, 0);             /* big-endian-flag */

#endif

    vnc_write_u8(vs, 1);             /* true-color-flag */

    if (vs->depth == 4) {

        vnc_write_u16(vs, 0xFF);     /* red-max */

        vnc_write_u16(vs, 0xFF);     /* green-max */

        vnc_write_u16(vs, 0xFF);     /* blue-max */

        vnc_write_u8(vs, 16);        /* red-shift */

        vnc_write_u8(vs, 8);         /* green-shift */

        vnc_write_u8(vs, 0);         /* blue-shift */

        vs->send_hextile_tile = send_hextile_tile_32;

    } else if (vs->depth == 2) {

        vnc_write_u16(vs, 31);       /* red-max */

        vnc_write_u16(vs, 63);       /* green-max */

        vnc_write_u16(vs, 31);       /* blue-max */

        vnc_write_u8(vs, 11);        /* red-shift */

        vnc_write_u8(vs, 5);         /* green-shift */

        vnc_write_u8(vs, 0);         /* blue-shift */

        vs->send_hextile_tile = send_hextile_tile_16;

    } else if (vs->depth == 1) {

        /* XXX: change QEMU pixel 8 bit pixel format to match the VNC one ? */

        vnc_write_u16(vs, 7);        /* red-max */

        vnc_write_u16(vs, 7);        /* green-max */

        vnc_write_u16(vs, 3);        /* blue-max */

        vnc_write_u8(vs, 5);         /* red-shift */

        vnc_write_u8(vs, 2);         /* green-shift */

        vnc_write_u8(vs, 0);         /* blue-shift */

        vs->send_hextile_tile = send_hextile_tile_8;

    }

    vs->client_red_max = vs->server_red_max;

    vs->client_green_max = vs->server_green_max;

    vs->client_blue_max = vs->server_blue_max;

    vs->client_red_shift = vs->server_red_shift;

    vs->client_green_shift = vs->server_green_shift;

    vs->client_blue_shift = vs->server_blue_shift;

    vs->pix_bpp = vs->depth * 8;

    vs->write_pixels = vnc_write_pixels_copy;



    vnc_write(vs, pad, 3);           /* padding */

}

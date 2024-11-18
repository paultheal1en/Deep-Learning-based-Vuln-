static int vnc_update_client(VncState *vs, int has_dirty)

{

    if (vs->need_update && vs->csock != -1) {

        VncDisplay *vd = vs->vd;

        int y;

        int n_rectangles;

        int saved_offset;

        int n;



        if (vs->output.offset && !vs->audio_cap && !vs->force_update)

            /* kernel send buffers are full -> drop frames to throttle */

            return 0;



        if (!has_dirty && !vs->audio_cap && !vs->force_update)

            return 0;



        /*

         * Send screen updates to the vnc client using the server

         * surface and server dirty map.  guest surface updates

         * happening in parallel don't disturb us, the next pass will

         * send them to the client.

         */

        n_rectangles = 0;

        vnc_write_u8(vs, VNC_MSG_SERVER_FRAMEBUFFER_UPDATE);

        vnc_write_u8(vs, 0);

        saved_offset = vs->output.offset;

        vnc_write_u16(vs, 0);



        for (y = 0; y < vd->server->height; y++) {

            int x;

            int last_x = -1;

            for (x = 0; x < vd->server->width / 16; x++) {

                if (vnc_get_bit(vs->dirty[y], x)) {

                    if (last_x == -1) {

                        last_x = x;

                    }

                    vnc_clear_bit(vs->dirty[y], x);

                } else {

                    if (last_x != -1) {

                        int h = find_and_clear_dirty_height(vs, y, last_x, x);

                        n = send_framebuffer_update(vs, last_x * 16, y,

                                                    (x - last_x) * 16, h);

                        n_rectangles += n;

                    }

                    last_x = -1;

                }

            }

            if (last_x != -1) {

                int h = find_and_clear_dirty_height(vs, y, last_x, x);

                n = send_framebuffer_update(vs, last_x * 16, y,

                                            (x - last_x) * 16, h);

                n_rectangles += n;

            }

        }

        vs->output.buffer[saved_offset] = (n_rectangles >> 8) & 0xFF;

        vs->output.buffer[saved_offset + 1] = n_rectangles & 0xFF;

        vnc_flush(vs);

        vs->force_update = 0;

        return n_rectangles;

    }



    if (vs->csock == -1)

        vnc_disconnect_finish(vs);



    return 0;

}

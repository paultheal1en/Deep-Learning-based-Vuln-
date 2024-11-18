static int send_rect_simple(VncState *vs, int x, int y, int w, int h)

{

    int max_size, max_width;

    int max_sub_width, max_sub_height;

    int dx, dy;

    int rw, rh;

    int n = 0;



    max_size = tight_conf[vs->tight_compression].max_rect_size;

    max_width = tight_conf[vs->tight_compression].max_rect_width;



    if (w > max_width || w * h > max_size) {

        max_sub_width = (w > max_width) ? max_width : w;

        max_sub_height = max_size / max_sub_width;



        for (dy = 0; dy < h; dy += max_sub_height) {

            for (dx = 0; dx < w; dx += max_width) {

                rw = MIN(max_sub_width, w - dx);

                rh = MIN(max_sub_height, h - dy);

                n += send_sub_rect(vs, x+dx, y+dy, rw, rh);

            }

        }

    } else {

        n += send_sub_rect(vs, x, y, w, h);

    }



    return n;

}

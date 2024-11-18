void vp8_mc(VP8Context *s, int luma,

            uint8_t *dst, uint8_t *src, const VP56mv *mv,

            int x_off, int y_off, int block_w, int block_h,

            int width, int height, int linesize,

            vp8_mc_func mc_func[3][3])

{

    if (AV_RN32A(mv)) {

        static const uint8_t idx[3][8] = {

            { 0, 1, 2, 1, 2, 1, 2, 1 }, // nr. of left extra pixels,

                                        // also function pointer index

            { 0, 3, 5, 3, 5, 3, 5, 3 }, // nr. of extra pixels required

            { 0, 2, 3, 2, 3, 2, 3, 2 }, // nr. of right extra pixels

        };

        int mx = (mv->x << luma)&7, mx_idx = idx[0][mx];

        int my = (mv->y << luma)&7, my_idx = idx[0][my];



        x_off += mv->x >> (3 - luma);

        y_off += mv->y >> (3 - luma);



        // edge emulation

        src += y_off * linesize + x_off;

        if (x_off < mx_idx || x_off >= width  - block_w - idx[2][mx] ||

            y_off < my_idx || y_off >= height - block_h - idx[2][my]) {

            s->dsp.emulated_edge_mc(s->edge_emu_buffer, src - my_idx * linesize - mx_idx, linesize,

                                block_w + idx[1][mx], block_h + idx[1][my],

                                x_off - mx_idx, y_off - my_idx, width, height);

            src = s->edge_emu_buffer + mx_idx + linesize * my_idx;

        }

        mc_func[my_idx][mx_idx](dst, linesize, src, linesize, block_h, mx, my);

    } else

        mc_func[0][0](dst, linesize, src + y_off * linesize + x_off, linesize, block_h, 0, 0);

}

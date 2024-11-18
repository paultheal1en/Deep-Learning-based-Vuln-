static int find_dirty_height(VncState *vs, int y, int last_x, int x)

{

    int h;



    for (h = 1; h < (vs->serverds.height - y); h++) {

        int tmp_x;

        if (!vnc_get_bit(vs->dirty_row[y + h], last_x))

            break;

        for (tmp_x = last_x; tmp_x < x; tmp_x++)

            vnc_clear_bit(vs->dirty_row[y + h], tmp_x);

    }



    return h;

}

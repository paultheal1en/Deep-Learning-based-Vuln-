static int check_intra_pred_mode(int mode, int mb_x, int mb_y)

{

    if (mode == DC_PRED8x8) {

        if (!(mb_x|mb_y))

            mode = DC_128_PRED8x8;

        else if (!mb_y)

            mode = LEFT_DC_PRED8x8;

        else if (!mb_x)

            mode = TOP_DC_PRED8x8;

    }

    return mode;

}

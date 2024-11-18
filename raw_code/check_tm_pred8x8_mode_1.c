int check_tm_pred8x8_mode(int mode, int mb_x, int mb_y)

{

    if (!mb_x)

        return mb_y ? VERT_PRED8x8 : DC_129_PRED8x8;

    else

        return mb_y ? mode : HOR_PRED8x8;

}

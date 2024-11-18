int check_intra_pred8x8_mode_emuedge(int mode, int mb_x, int mb_y)

{

    switch (mode) {

    case DC_PRED8x8:

        return check_dc_pred8x8_mode(mode, mb_x, mb_y);

    case VERT_PRED8x8:

        return !mb_y ? DC_127_PRED8x8 : mode;

    case HOR_PRED8x8:

        return !mb_x ? DC_129_PRED8x8 : mode;

    case PLANE_PRED8x8: /* TM */

        return check_tm_pred8x8_mode(mode, mb_x, mb_y);

    }

    return mode;

}
